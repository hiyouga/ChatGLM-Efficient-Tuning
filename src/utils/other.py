import os
import sys
import json
import torch
import logging
from typing import Dict, List, Optional

from transformers import Seq2SeqTrainingArguments
from transformers.trainer import TRAINER_STATE_NAME
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor


from peft.utils.other import WEIGHTS_NAME


IGNORE_INDEX = -100
VALUE_HEAD_FILE_NAME = "value_head.bin"
FINETUNING_ARGS_NAME = "finetuning_args.bin"
PREDICTION_FILE_NAME = "generated_predictions.txt"


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class AverageMeter:
    r"""
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Avoid runtime error in model.generate(do_sample=True).
# Borrowed from: https://huggingface.co/THUDM/chatglm-6b/blob/658202d88ac4bb782b99e99ac3adff58b4d0b813/modeling_chatglm.py#L54
class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    return logits_processor


# Includes: (1) cast the layernorm in fp32 (2) make output embedding layer require grads (3) upcast the lm_head to fp32
# Inspired by: https://github.com/huggingface/peft/blob/c0209c35abbf88c63aa267800d98a8e212ed0a42/src/peft/utils/other.py#L35
def prepare_model_for_training(
        model: PreTrainedModel,
        output_embedding_layer_name: Optional[str] = "lm_head",
        use_gradient_checkpointing: Optional[bool] = True,
        layer_norm_names: Optional[List[str]] = ["layernorm"] # for chatglm setting
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


def filter_model_params(model: torch.nn.Module) -> Dict[str, torch.Tensor]: # filter out freezed parameters
    state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in model.named_parameters():
        if v.requires_grad:
            filtered_state_dict[k] = state_dict[k]
    return filtered_state_dict


def save_trainable_params(save_directory: os.PathLike, model: torch.nn.Module) -> None:
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file.")
    os.makedirs(save_directory, exist_ok=True)
    filtered_state_dict = filter_model_params(model)
    torch.save(filtered_state_dict, os.path.join(save_directory, WEIGHTS_NAME))


def load_trainable_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> None:
    weights_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if not os.path.exists(weights_file):
        raise ValueError(f"Provided path ({checkpoint_dir}) does not contain the pretrained weights.")
    model_state_dict = torch.load(weights_file)
    model.load_state_dict(model_state_dict, strict=False) # skip missing keys


def save_valuehead_params(save_directory: os.PathLike, v_head: torch.nn.Module) -> None:
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file.")
    os.makedirs(save_directory, exist_ok=True)
    torch.save(v_head.state_dict(), os.path.join(save_directory, VALUE_HEAD_FILE_NAME))


def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> None:
    valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
    if not os.path.exists(valuehead_file):
        raise ValueError(f"Provided path ({checkpoint_dir}) does not contain the valuehead weights.")
    valuehead_state_dict = torch.load(valuehead_file)
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
    model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))


def smooth(scalars: List[float], weight: Optional[float] = 0.95) -> List[float]:
    """
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(training_args: Seq2SeqTrainingArguments, keys: Optional[List[str]] = ["loss"]) -> None:
    import matplotlib.pyplot as plt
    data = json.load(open(os.path.join(training_args.output_dir, TRAINER_STATE_NAME), "r"))

    for key in keys:
        steps, metrics = [], []

        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])
        smoothed_value = smooth(metrics)

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4)
        plt.plot(steps, smoothed_value)
        plt.title("training {} of {}".format(key, training_args.output_dir))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.savefig(os.path.join(training_args.output_dir, "training_{}.jpg".format(key)), format="jpg", dpi=100)
        print("Figure saved:", os.path.join(training_args.output_dir, "training_{}.jpg".format(key)))
