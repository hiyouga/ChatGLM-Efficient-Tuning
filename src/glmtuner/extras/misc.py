import torch
from typing import Dict, List, Optional

from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor

from glmtuner.extras.constants import LAYERNORM_NAMES


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


# Includes: (1) cast the layernorm in fp32 (2) make output embedding layer require grads (3) upcast the lm_head to fp32
# Inspired by: https://github.com/huggingface/peft/blob/c0209c35abbf88c63aa267800d98a8e212ed0a42/src/peft/utils/other.py#L35
def prepare_model_for_training(
    model: PreTrainedModel,
    finetuning_type: str,
    output_embedding_base_layer: torch.nn.Module,
    output_embedding_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layer_norm_names: Optional[List[str]] = LAYERNORM_NAMES
) -> PreTrainedModel:

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if finetuning_type != "full" and hasattr(output_embedding_base_layer, output_embedding_layer_name):
        output_embedding_layer = getattr(output_embedding_base_layer, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(output_embedding_base_layer, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model


def auto_configure_device_map(num_gpus: int, use_v2: bool) -> Dict[str, int]:
    r"""
    Configures device map for ChatGLM.

    Borrowed from: https://github.com/THUDM/ChatGLM-6B/blob/dev_multi_gpu/utils.py#L8
    """
    num_layers = 28
    layers_per_gpu = 30 / num_gpus
    if use_v2:
        device_map = {
            "transformer.embedding.word_embeddings": 0,
            "transformer.encoder.final_layernorm": 0,
            "transformer.output_layer": 0,
            "transformer.rotary_pos_emb": 0,
            "transformer.prefix_encoder": 0,
            "lm_head": 0
        }
    else:
        device_map = {
            "transformer.word_embeddings": 0,
            "transformer.final_layernorm": 0,
            "transformer.prefix_encoder": 0,
            "lm_head": 0
        }

    added_layers = 2
    target_gpu = 0

    for i in range(num_layers):
        if added_layers >= layers_per_gpu:
            target_gpu += 1
            added_layers = 0
        assert target_gpu < num_gpus
        if use_v2:
            device_map[f"transformer.encoder.layers.{i}"] = target_gpu
        else:
            device_map[f"transformer.layers.{i}"] = target_gpu
        added_layers += 1

    return device_map


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
