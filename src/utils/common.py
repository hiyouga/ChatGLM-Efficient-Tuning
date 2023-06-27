import os
import sys
import torch
import hashlib
from types import MethodType
from typing import List, Literal, Optional, Tuple

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

import datasets
from datasets import Dataset, concatenate_datasets, load_dataset

from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)

from peft.utils import CONFIG_NAME, WEIGHTS_NAME

from trl import AutoModelForCausalLMWithValueHead

from .config import (
    ModelArguments,
    DataTrainingArguments,
    FinetuningArguments,
    GeneratingArguments
)

from .other import (
    get_logger,
    load_trainable_params,
    load_valuehead_params,
    print_trainable_params,
    prepare_model_for_training,
    IGNORE_INDEX
)

check_min_version("4.27.4")
require_version("datasets>=2.10.0", "To fix: pip install datasets>=2.10.0")
require_version("accelerate>=0.19.0", "To fix: pip install accelerate>=0.19.0")
require_version("peft>=0.3.0", "To fix: pip install peft>=0.3.0")
require_version("trl>=0.4.4", "To fix: pip install trl>=0.4.4")


logger = get_logger(__name__)


def init_adapter(
        model: PreTrainedModel,
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: bool
) -> PreTrainedModel:
    r"""
    Initializes the adapters.

    Support full-parameter, freeze, P-Tuning v2 and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full":
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze":
        logger.info("Fine-tuning method: Freeze")
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in finetuning_args.trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    if finetuning_args.finetuning_type == "p_tuning":
        logger.info("Fine-tuning method: P-Tuning v2") # nothing to do

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            assert len(model_args.checkpoint_dir) == 1, "Only LoRA tuning accepts multiple checkpoints."
            assert load_trainable_params(model, model_args.checkpoint_dir[0]), "Model checkpoint is not correctly loaded."
        else:
            assert len(model_args.checkpoint_dir) == 1, "Quantized model only accepts a single checkpoint."

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        lastest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], WEIGHTS_NAME)), \
                "Provided path ({}) does not contain a LoRA weight.".format(model_args.checkpoint_dir[0])
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], CONFIG_NAME)), \
                "The given checkpoint may be not a LoRA checkpoint, please specify `--finetuning_type full/p_tuning/freeze` instead."

            if is_trainable and model_args.resume_lora_training: # continually train on the lora weights
                checkpoints_to_merge, lastest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if lastest_checkpoint is not None: # resume lora training
                model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=True)

        if is_trainable and lastest_checkpoint is None: # create new lora weights while training
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, # we should regard ChatGLM as a causal LM
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model


def load_pretrained(
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

    # P-Tuning v2 configurations.
    # We use the built-in p-tuning method of ChatGLM, we cannot use PEFT since the attention masks of ChatGLM are unusual. >_<
    if finetuning_args.finetuning_type == "p_tuning":
        assert not model_args.use_v2, "ChatGLM2-6B does not support P-Tuning v2."
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

    # Load and prepare pretrained models (without valuehead).
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, **config_kwargs)

    if model_args.use_v2:
        def get_input_embeddings(self):
            return self.transformer.embedding
        model.get_input_embeddings = MethodType(get_input_embeddings, model)
        model.lm_head = model.transformer.output_layer # need fix: cast to float

    model = prepare_model_for_training(model, finetuning_args.finetuning_type) if is_trainable else model
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


def prepare_args(
        stage: Literal["sft", "rm", "ppo"]
) -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # Provide arguments with a json file.
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Check arguments (do not check finetuning_args since it may be loaded from checkpoints)
    if stage != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True at RM and PPO stages.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True while training.")

    if training_args.do_predict and (not training_args.predict_with_generate):
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type == "full":
            raise ValueError("Quantization is incompatible with the full-parameter tuning.")

        if finetuning_args.finetuning_type == "p_tuning" and training_args.fp16:
            raise ValueError("FP16 training conflicts with quantized P-Tuning.")

        if not training_args.do_train:
            logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if training_args.do_train and (not training_args.fp16):
        logger.warning("We recommend enable fp16 mixed precision training for ChatGLM-6B.")

    if training_args.local_rank != -1 and training_args.ddp_find_unused_parameters is None:
        logger.warning("`ddp_find_unused_parameters` needs to be set as False in DDP training.")
        training_args.ddp_find_unused_parameters = False

    training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim # suppress warning

    if model_args.quantization_bit is not None:
        if training_args.fp16:
            model_args.compute_dtype = torch.float16
        elif training_args.bf16:
            model_args.compute_dtype = torch.bfloat16
        else:
            model_args.compute_dtype = torch.float32

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"  distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args


def prepare_infer_args() -> Tuple[ModelArguments, FinetuningArguments, GeneratingArguments]:

    parser = HfArgumentParser((ModelArguments, FinetuningArguments, GeneratingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # Provide arguments with a json file.
        model_args, finetuning_args, generating_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, finetuning_args, generating_args = parser.parse_args_into_dataclasses()

    return model_args, finetuning_args, generating_args


def prepare_data(
        model_args: ModelArguments,
        data_args: DataTrainingArguments
) -> Dataset:

    def checksum(file_path, hash):
        with open(file_path, "rb") as datafile:
            binary_data = datafile.read()
        sha1 = hashlib.sha1(binary_data).hexdigest()
        if sha1 != hash:
            logger.warning("Checksum failed for {}. It may vary depending on the platform.".format(file_path))

    ext2type = {
        "csv": "csv",
        "json": "json",
        "jsonl": "json"
    }

    max_samples = data_args.max_samples
    all_datasets: List[Dataset] = [] # support multiple datasets

    for dataset_attr in data_args.dataset_list:

        logger.info("Loading dataset {}...".format(dataset_attr))

        if dataset_attr.load_from == "hf_hub":
            data_path = dataset_attr.dataset_name
            data_files = None
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            data_files = None
        elif dataset_attr.load_from == "file":
            data_path = None
            data_files: List[str] = []

            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))

                    if data_path is None:
                        data_path = ext2type.get(data_files[0].split(".")[-1], None)
                    else:
                        assert ext2type.get(data_files[-1].split(".")[-1], None) == data_path, "file type does not match."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = ext2type.get(data_files[0].split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be csv, json or jsonl."

            if len(data_files) == 1 and dataset_attr.dataset_sha1 is not None:
                checksum(data_files[0], dataset_attr.dataset_sha1)
            else:
                logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json or too many files.")
        else:
            raise NotImplementedError

        raw_datasets = load_dataset(
            data_path,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None
        )
        dataset = raw_datasets[data_args.split]

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        dummy_data = [None] * len(dataset)
        for column_name, target_name in [
            ("prompt_column", "prompt"),
            ("query_column", "query"),
            ("response_column", "response"),
            ("history_column", "history")
        ]: # every dataset will have 4 columns same as each other
            if getattr(dataset_attr, column_name) != target_name:
                if getattr(dataset_attr, column_name):
                    dataset = dataset.rename_column(getattr(dataset_attr, column_name), target_name)
                else: # None or empty string
                    dataset = dataset.add_column(target_name, dummy_data)
        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        all_datasets = all_datasets[0]
    else:
        all_datasets = concatenate_datasets(all_datasets)

    return all_datasets


def preprocess_data(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_args: DataTrainingArguments,
        training_args: Seq2SeqTrainingArguments,
        stage: Literal["sft", "rm", "ppo"]
) -> Dataset:

    column_names = list(dataset.column_names)
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    def format_example(examples): # support question with a single answer or multiple answers
        for i in range(len(examples["prompt"])):
            if examples["prompt"][i] and examples["response"][i]:
                query, answer = examples["prompt"][i], examples["response"][i]
                query = query + examples["query"][i] if examples["query"][i] else query
                history = examples["history"][i] if examples["history"][i] else []
                prompt = ""
                for j, (old_query, response) in enumerate(history):
                    prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(j+1, old_query, response)
                prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history)+1, query)
                prompt = prefix + prompt
                yield prompt, answer

    def preprocess_supervised_dataset(examples):
        # v1: build inputs with format `X [gMASK] <sop> Y <eop>` and labels with format `[IGNORE] ... [IGNORE] Y <eop>`
        # v2: build inputs with format `[gMASK] sop X Y </s>` and labels with format `[IGNORE] ... [IGNORE] Y </s>`
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 2: # gmask and sop tokens
                source_ids = source_ids[:data_args.max_source_length - 2]
            if len(target_ids) > data_args.max_target_length - 1: # eos token
                target_ids = target_ids[:data_args.max_target_length - 1]

            context_length = len(source_ids) + 2 # gmask and sop tokens
            input_ids = tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)
            labels = [IGNORE_INDEX] * context_length + input_ids[context_length:]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_evaluation_dataset(examples):
        # v1: build inputs with format `X [gMASK] <sop>` and labels with format `Y [gMASK] <sop>`
        # v2: build inputs with format `[gMASK] sop X` and labels with format `[gMASK] sop Y`
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 2: # gmask and sop tokens
                source_ids = source_ids[:data_args.max_source_length - 2]
            if len(target_ids) > data_args.max_target_length - 2: # gmask and sop tokens
                target_ids = target_ids[:data_args.max_target_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(source_ids)
            labels = tokenizer.build_inputs_with_special_tokens(target_ids)

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_pairwise_dataset(examples):
        # v1: build input pairs with format `X [gMASK] <sop> Y1 <eop>` and `X [gMASK] <sop> Y2 <eop>`
        # v2: build input pairs with format `[gMASK] sop X Y1 </s>` and `[gMASK] sop X Y2 </s>`
        model_inputs = {"accept_ids": [], "reject_ids": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            accept_ids = tokenizer.encode(text=answer[0], add_special_tokens=False)
            reject_ids = tokenizer.encode(text=answer[1], add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 2: # gmask and sop tokens
                source_ids = source_ids[:data_args.max_source_length - 2]
            if len(accept_ids) > data_args.max_target_length - 1: # eos token
                accept_ids = accept_ids[:data_args.max_target_length - 1]
            if len(reject_ids) > data_args.max_target_length - 1: # eos token
                reject_ids = reject_ids[:data_args.max_target_length - 1]

            accept_ids = tokenizer.build_inputs_with_special_tokens(source_ids[:], accept_ids) # avoid copying error
            reject_ids = tokenizer.build_inputs_with_special_tokens(source_ids[:], reject_ids)

            model_inputs["accept_ids"].append(accept_ids)
            model_inputs["reject_ids"].append(reject_ids)
        return model_inputs

    def print_sft_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))

    def print_pairwise_dataset_example(example):
        print("accept_ids:\n{}".format(example["accept_ids"]))
        print("accepts:\n{}".format(tokenizer.decode(example["accept_ids"], skip_special_tokens=False)))
        print("reject_ids:\n{}".format(example["reject_ids"]))
        print("rejects:\n{}".format(tokenizer.decode(example["reject_ids"], skip_special_tokens=False)))

    def print_ppo_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if stage == "sft":
        preprocess_function = preprocess_evaluation_dataset \
            if training_args.predict_with_generate else preprocess_supervised_dataset
    elif stage == "rm":
        preprocess_function = preprocess_pairwise_dataset
    elif stage == "ppo":
        preprocess_function = preprocess_evaluation_dataset

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )

        if stage == "sft":
            print_sft_dataset_example(dataset[0])
        elif stage == "rm":
            print_pairwise_dataset_example(dataset[0])
        elif stage == "ppo":
            print_ppo_dataset_example(dataset[0])

        return dataset
