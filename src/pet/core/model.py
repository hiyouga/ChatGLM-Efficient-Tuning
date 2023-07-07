import os
import torch
from typing import Literal, Optional, Tuple

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from trl import AutoModelForCausalLMWithValueHead

from extras.logging import get_logger
from extras.misc import prepare_model_for_training, print_trainable_params
from extras.save_and_load import load_valuehead_params
from hparams import ModelArguments, FinetuningArguments
from pet.core.adapter import init_adapter


logger = get_logger(__name__)


check_min_version("4.27.4")
require_version("datasets>=2.10.0", "To fix: pip install datasets>=2.10.0")
require_version("accelerate>=0.19.0", "To fix: pip install accelerate>=0.19.0")
require_version("peft>=0.3.0", "To fix: pip install peft>=0.3.0")
require_version("trl>=0.4.4", "To fix: pip install trl>=0.4.4")


def load_model_and_tokenizer(
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: Optional[bool] = False,
        stage: Optional[Literal["sft", "rm", "ppo"]] = "sft"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    assert stage == "sft" or finetuning_args.finetuning_type == "lora", \
        "RM and PPO training can only be performed with LoRA method."

    quantization = None
    if model_args.quantization_bit is not None:
        if is_trainable:
            if finetuning_args.finetuning_type == "full":
                raise ValueError("Full-parameter fine-tuning does not support quantization.")
            elif finetuning_args.finetuning_type == "p_tuning":
                quantization = "cpm" # use cpm's quantization
            else:
                quantization = "bnb" # use bnb's quantization
        else:
            quantization = "cpm"

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
        **config_kwargs
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        **config_kwargs
    )

    # P-Tuning v2 configurations. Use the built-in p-tuning method of ChatGLM.
    if finetuning_args.finetuning_type == "p_tuning":
        config.pre_seq_len = finetuning_args.pre_seq_len # enable this will fix other parameters automatically
        config.prefix_projection = finetuning_args.prefix_projection

    # Quantization configurations for Full, Freeze and LoRA in training (using bitsandbytes library).
    if quantization == "bnb":
        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("transformers>=4.30.1", "To fix: pip install transformers>=4.30.1")
            require_version("accelerate>=0.20.3", "To fix: pip install accelerate>=0.20.3")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if model_args.checkpoint_dir is not None and finetuning_args.finetuning_type == "full":
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    # Load and prepare pretrained models (without valuehead).
    model = AutoModel.from_pretrained(model_to_load, config=config, **config_kwargs)

    # Register auto class to save the custom code files.
    if hasattr(config, "auto_map") and "AutoConfig" in config.auto_map:
        config.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
        tokenizer.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoModel" in config.auto_map:
        model.__class__.register_for_auto_class()

    if model_args.use_v2:
        assert tokenizer.eos_token_id is not None, "Please update the *.json and *.py files of ChatGLM2-6B from HuggingFace."
        model.lm_head = model.transformer.output_layer
        output_embedding_base_layer = model.transformer
        output_embedding_layer_name = "output_layer"
    else:
        assert tokenizer.eos_token_id == 130005, "Please specify `use_v2` argument while using ChatGLM2-6B."
        output_embedding_base_layer = model
        output_embedding_layer_name = "lm_head"

    # Initialize adapters
    model = prepare_model_for_training(
        model,
        finetuning_args.finetuning_type,
        output_embedding_base_layer,
        output_embedding_layer_name
    ) if is_trainable else model
    model = init_adapter(model, model_args, finetuning_args, is_trainable)

    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.half() # cast all params to float16 for inference

    # Quantization with the built-in method for P-Tuning v2 training or evaluation.
    # Model parameters should be cast to float16 in quantized P-Tuning setting.
    if quantization == "cpm":
        if is_trainable: # convert all params into half precision except prefix_encoder in training
            for name, param in model.named_parameters():
                if "prefix_encoder" not in name:
                    param.data = param.data.to(torch.float16)

        model.quantize(model_args.quantization_bit) # built-in method in ChatGLM-6B, also an in-place operation

    if quantization is not None:
        logger.info("Quantized model to {} bit.".format(model_args.quantization_bit))

    if stage == "rm" or stage == "ppo": # add value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        if stage == "rm" and model_args.checkpoint_dir is not None: # load valuehead weights to evaluate reward model
            logger.warning("Only the last checkpoint containing valuehead will be loaded as the valuehead.")
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo": # load reward model
            assert is_trainable, "PPO stage cannot be performed at evaluation."
            assert model_args.reward_model is not None, "Reward model is necessary for PPO training."
            logger.info("Load reward model from {}".format(model_args.reward_model))
            model.pretrained_model.load_adapter(model_args.reward_model, "reward", is_trainable=False)
            assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."

    print_trainable_params(model)

    return model, tokenizer
