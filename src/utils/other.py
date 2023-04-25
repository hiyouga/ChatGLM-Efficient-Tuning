import os
import sys
import json
import torch
import logging
from typing import Dict, List, Optional

from transformers import Seq2SeqTrainingArguments
from transformers.trainer import TRAINER_STATE_NAME
from transformers.modeling_utils import PreTrainedModel

from peft.utils.other import WEIGHTS_NAME, CONFIG_NAME


logger = logging.getLogger(__name__) # setup logging
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
                trainable_params, all_param, 100 * trainable_params / all_param))


def filter_model_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]: # filter out the freezed parameters
    state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in model.named_parameters():
        if v.requires_grad:
            filtered_state_dict[k] = state_dict[k]
    return filtered_state_dict


def save_trainable_params(save_directory: os.PathLike, model: torch.nn.Module) -> None:
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
    os.makedirs(save_directory, exist_ok=True)
    filtered_state_dict = filter_model_params(model)
    torch.save(filtered_state_dict, os.path.join(save_directory, WEIGHTS_NAME))


def load_trainable_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> None:
    model_state_dict = torch.load(os.path.join(checkpoint_dir, WEIGHTS_NAME))
    model.load_state_dict(model_state_dict, strict=False) # skip missing keys


# This function includes: (1) cast the layernorm in fp32 (2) make output embedding layer require grads (3) upcast the lm_head to fp32
# Inspired by: https://github.com/huggingface/peft/blob/c0209c35abbf88c63aa267800d98a8e212ed0a42/src/peft/utils/other.py#L35
def prepare_model_for_training(
        model: PreTrainedModel,
        output_embedding_layer_name: Optional[str] = "lm_head",
        use_gradient_checkpointing: Optional[bool] = True,
        layer_norm_names: List[str] = ["layernorm"] # for chatglm setting
) -> PreTrainedModel:
    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model


# This function merges lora weights from multiple checkpoints
# Inspired by: https://github.com/huggingface/peft/blob/34027fe813756897767b9a6f19ae7f1c4c7b418c/src/peft/tuners/lora.py#L451
def merge_lora_weights(model: PreTrainedModel, checkpoints_to_merge: List[str]) -> int:
    checkpoint_merged = 0
    for checkpoint_dir in checkpoints_to_merge:
        adapter_config = json.load(open(os.path.join(checkpoint_dir, CONFIG_NAME), "r"))
        adapter_model = torch.load(os.path.join(checkpoint_dir, WEIGHTS_NAME))
        scaling = adapter_config["lora_alpha"] / adapter_config["r"]
        is_merged = False
        for name, param in model.named_parameters():
            if "weight" not in name: # skip bias
                continue
            lora_a_name = "base_model.model." + ".".join(name.split(".")[:-1]) + ".lora_A.weight"
            lora_b_name = "base_model.model." + ".".join(name.split(".")[:-1]) + ".lora_B.weight"
            lora_a_weight, lora_b_weight = None, None
            for adapter_name, adapter_param in adapter_model.items():
                if adapter_name == lora_a_name:
                    lora_a_weight = adapter_param
                if adapter_name == lora_b_name:
                    lora_b_weight = adapter_param
            if lora_a_weight is not None and lora_b_weight is not None:
                weight_to_merge = lora_b_weight @ lora_a_weight
                weight_to_merge = weight_to_merge.T if adapter_config["fan_in_fan_out"] else weight_to_merge
                param.data += weight_to_merge.to(param.device) * scaling
                is_merged = True
        checkpoint_merged = checkpoint_merged + 1 if is_merged else checkpoint_merged
    return checkpoint_merged


def plot_loss(training_args: Seq2SeqTrainingArguments) -> None:
    import matplotlib.pyplot as plt
    FIGURE_NAME = "trainer_state.png"
    data = json.load(open(os.path.join(training_args.output_dir, TRAINER_STATE_NAME), "r"))
    train_steps, train_losses = [], []
    for i in range(len(data["log_history"]) - 1):
        train_steps.append(data["log_history"][i]["step"])
        train_losses.append(data["log_history"][i]["loss"])
    plt.figure()
    plt.plot(train_steps, train_losses)
    plt.title("training loss of {}".format(training_args.output_dir))
    plt.xlabel("step")
    plt.ylabel("training loss")
    plt.savefig(os.path.join(training_args.output_dir, FIGURE_NAME), format="png", transparent=True, dpi=300)
    print("Figure saved: {}".format(os.path.join(training_args.output_dir, FIGURE_NAME)))


IGNORE_INDEX = -100
FINETUNING_ARGS_NAME = "finetuning_args.bin"
PREDICTION_FILE_NAME = "generated_predictions.txt"
