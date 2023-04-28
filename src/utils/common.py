import os
import sys
import torch
import hashlib
import logging
from typing import Literal, Optional, Tuple

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed
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

from trl import AutoModelForCausalLMWithValueHead

from .config import (
    ModelArguments,
    DataTrainingArguments,
    FinetuningArguments
)

from .other import (
    load_trainable_params,
    print_trainable_params,
    prepare_model_for_training,
    merge_lora_weights,
    IGNORE_INDEX,
    FINETUNING_ARGS_NAME
)


logger = logging.getLogger(__name__) # setup logging
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


check_min_version("4.27.4")
require_version("datasets>=2.10.0", "To fix: pip install datasets>=2.10.0")
require_version("peft>=0.3.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")


def init_adapter(
        model: PreTrainedModel,
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: bool
) -> None:
    r"""
    Initializes the adapters.

    Note that the trainable parameters must be cast to float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "freeze":
        logger.info("Fine-tuning method: Freeze")
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in finetuning_args.trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

        if model_args.checkpoint_dir is not None: # freeze only accepts a single checkpoint
            load_trainable_params(model, model_args.checkpoint_dir[0])

    if finetuning_args.finetuning_type == "p_tuning":
        logger.info("Fine-tuning method: P-Tuning v2")
        model.transformer.prefix_encoder.float() # other parameters are already fixed

        if model_args.checkpoint_dir is not None: # p-tuning v2 only accepts a single checkpoint
            load_trainable_params(model, model_args.checkpoint_dir[0])

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
        lastest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            if finetuning_args.resume_lora_training: # continually training on the lora weights
                checkpoints_to_merge, lastest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            if len(checkpoints_to_merge) != 0 and loaded_in_8bit:
                raise ValueError("8-bit model does not support merging the LoRA weights.")

            checkpoint_merged = merge_lora_weights(model, checkpoints_to_merge)

            if lastest_checkpoint is not None: # resume lora training
                model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=is_trainable)
                if not is_trainable:
                    model.merge_and_unload()
                    checkpoint_merged += 1
            logger.info("Merged {} model checkpoint(s).".format(checkpoint_merged))

        if lastest_checkpoint is None: # create new lora weights
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, lora_config)

    if not is_trainable:
        for param in model.parameters():
            param.requires_grad_(False) # fix all params
            param.data = param.data.to(torch.float16) # cast all params to float16

    return model


def load_pretrained(
        model_args: ModelArguments,
        training_args: Optional[Seq2SeqTrainingArguments] = None,
        finetuning_args: Optional[FinetuningArguments] = None,
        is_trainable: Optional[bool] = False,
        stage: Optional[Literal["sft", "rwd", "ppo"]] = "sft"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Load pretrained model and tokenizer.
    """

    if (not is_trainable) and (model_args.checkpoint_dir is None):
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    if model_args.checkpoint_dir is not None: # load fine-tuned model from checkpoint
        for checkpoint_dir in model_args.checkpoint_dir:
            if not os.path.isfile(os.path.join(checkpoint_dir, FINETUNING_ARGS_NAME)):
                raise ValueError("The fine-tuning arguments are not found in the provided dictionary.")
        logger.info("Load fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))
        finetuning_args = torch.load(os.path.join(model_args.checkpoint_dir[0], FINETUNING_ARGS_NAME))

    quantization = None
    if model_args.quantization_bit is not None:
        if is_trainable:
            if finetuning_args.finetuning_type != "p_tuning":
                quantization = "hf" # use huggingface's quantization
            else:
                quantization = "cpm" # use cpm's quantization
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
        **config_kwargs
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        **config_kwargs
    )

    # P-Tuning v2 configurations.
    # We use the built-in p-tuning method of ChatGLM, we cannot use PEFT since the attention masks of ChatGLM are unusual. >_<
    if finetuning_args.finetuning_type == "p_tuning":
        config.pre_seq_len = finetuning_args.pre_seq_len # enable this will fix other parameters automatically
        config.prefix_projection = finetuning_args.prefix_projection

    # Quantization configurations for Freeze and LoRA in training (using bitsandbytes library).
    if quantization == "hf":
        if model_args.quantization_bit != 8:
            raise ValueError("Freeze and LoRA fine-tuning only accept 8-bit quantization.")
        require_version("bitsandbytes>=0.37.0", "bitsandbytes library is required to use this feature.")
        from bitsandbytes.cuda_setup.main import get_compute_capability, get_cuda_lib_handle, is_cublasLt_compatible
        cuda = get_cuda_lib_handle()
        cc = get_compute_capability(cuda)
        if not is_cublasLt_compatible(cc):
            raise ValueError("The current GPU(s) is incompatible with quantization.")
        config_kwargs["load_in_8bit"] = True
        config_kwargs["device_map"] = "auto" # it should not be specified outside of load_in_8bit

    # Load and prepare pretrained models (without valuehead).
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, **config_kwargs)
    model = prepare_model_for_training(model) if is_trainable else model
    model = init_adapter(model, model_args, finetuning_args, is_trainable)

    # Quantization with the built-in method for P-Tuning v2 training or evaluation.
    # Model parameters should be cast to float16 in quantized P-Tuning setting.
    if quantization == "cpm":
        if model_args.quantization_bit != 4 and model_args.quantization_bit != 8:
            raise ValueError("P-Tuning v2 and inference modes only accept 4-bit or 8-bit quantization.")
        if is_trainable and training_args.fp16:
            raise ValueError("FP16 training conflicts with cpm quantization.")
        model = model.quantize(model_args.quantization_bit).half()

    if quantization is not None:
        logger.info("Quantized model to {} bit.".format(model_args.quantization_bit))

    if stage != "sft":
        model = AutoModelForCausalLMWithValueHead(model)

    print_trainable_params(model)

    return model, tokenizer


def prepare_args() -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Provide arguments with a json file.
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    # Check arguments (do not check finetuning_args since it may be loaded from checkpoints)
    if int(training_args.do_train) + int(training_args.do_eval) + int(training_args.do_predict) != 1:
        raise ValueError("We must perform single operation among do_train, do_eval and do_predict.")

    if model_args.quantization_bit is not None and training_args.do_train == False:
        logger.warning("We do not recommend to evaluaute model in 4/8-bit mode.")

    if not training_args.fp16:
        logger.warning("We recommend enable fp16 mixed precision training for ChatGLM-6B.")

    training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim # suppress warning

    # Set logger
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"  distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args


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

    max_samples = data_args.max_samples
    all_datasets = [] # support multiple datasets

    for dataset_info in data_args.dataset_list:

        logger.info("Loading dataset {}...".format(dataset_info))

        if dataset_info.load_from == "hf_hub":
            raw_datasets = load_dataset(dataset_info.dataset_name, cache_dir=model_args.cache_dir)
        elif dataset_info.load_from == "script":
            raw_datasets = load_dataset(
                os.path.join(data_args.dataset_dir, dataset_info.dataset_name),
                cache_dir=model_args.cache_dir
            )
        elif dataset_info.load_from == "file":
            data_file = os.path.join(data_args.dataset_dir, dataset_info.file_name) # support json, jsonl and csv
            extension = dataset_info.file_name.split(".")[-1]

            if dataset_info.file_sha1 is not None:
                checksum(data_file, dataset_info.file_sha1)
            else:
                logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.")

            raw_datasets = load_dataset(
                extension,
                data_files=data_file,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None
            )
        else:
            raise NotImplementedError

        dataset = raw_datasets[data_args.split]

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        dummy_data = [None] * len(dataset)
        for column, column_name in [
            ("prompt_column", "prompt"),
            ("query_column", "query"),
            ("response_column", "response"),
            ("history_column", "history")
        ]: # every dataset will have 4 columns same as each other
            if getattr(dataset_info, column) != column_name:
                if getattr(dataset_info, column):
                    dataset = dataset.rename_column(getattr(dataset_info, column), column_name)
                else: # None or empty string
                    dataset = dataset.add_column(column_name, dummy_data)
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
        stage: Optional[Literal["sft", "rwd", "ppo"]] = "sft"
) -> Dataset:

    column_names = list(dataset.column_names)
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    def format_example(examples): # support question with a single answer or multiple answers
        for i in range(len(examples["prompt"])):
            if examples["prompt"][i] and examples["response"][i]:
                query, answer = examples["prompt"][i], examples["response"][i]
                if examples["query"][i]:
                    query += examples["query"][i]
                if examples["history"][i]:
                    prompt = ""
                    history = examples["history"][i]
                    for i, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                else:
                    prompt = query
                prompt = prefix + prompt
                yield prompt, answer

    def preprocess_function_train(examples):
        # build inputs with format `X [gMASK] [BOS] Y [EOS]` and labels with format `[IGNORE] ... [IGNORE] [BOS] Y [EOS]`
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 1: # gmask token
                source_ids = source_ids[:data_args.max_source_length - 1]
            if len(target_ids) > data_args.max_target_length - 2: # bos and eos tokens
                target_ids = target_ids[:data_args.max_target_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            labels = [IGNORE_INDEX] * context_length + input_ids[context_length:]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_function_eval(examples):
        # build inputs with format `[PAD] ... [PAD] X [gMASK] [BOS]` and labels with format `Y [gMASK] [BOS]`
        # left-padding is needed for prediction, use the built-in function of the tokenizer
        inputs, targets = [], []
        for prompt, answer in format_example(examples):
            inputs.append(prompt)
            targets.append(answer)
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
        labels = tokenizer(text_target=targets, max_length=data_args.max_target_length, truncation=True)
        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l_id if l_id != tokenizer.pad_token_id else IGNORE_INDEX) for l_id in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function_train_pair(examples):
        # build input pairs with format `X [gMASK] [BOS] Y [EOS]` and `X [gMASK] [BOS] Y [EOS]`
        model_inputs = {"accept_ids": [], "reject_ids": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            accept_ids = tokenizer.encode(text=answer[0], add_special_tokens=False)
            reject_ids = tokenizer.encode(text=answer[1], add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 1:
                source_ids = source_ids[:data_args.max_source_length - 1]
            if len(accept_ids) > data_args.max_target_length - 2:
                accept_ids = accept_ids[:data_args.max_target_length - 2]
            if len(reject_ids) > data_args.max_target_length - 2:
                reject_ids = reject_ids[:data_args.max_target_length - 2]

            accept_ids = tokenizer.build_inputs_with_special_tokens(source_ids, accept_ids)
            reject_ids = tokenizer.build_inputs_with_special_tokens(source_ids, reject_ids)

            model_inputs["accept_ids"].append(accept_ids)
            model_inputs["reject_ids"].append(reject_ids)
        return model_inputs

    def print_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"])))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(tokenizer.decode(example["labels"])))

    def print_pairwise_dataset_example(example):
        print("accept_ids:\n{}".format(example["accept_ids"]))
        print("accepts:\n{}".format(tokenizer.decode(example["accept_ids"])))
        print("reject_ids:\n{}".format(example["reject_ids"]))
        print("rejects:\n{}".format(tokenizer.decode(example["reject_ids"])))

    if stage == "sft":
        preprocess_function = preprocess_function_train if training_args.do_train else preprocess_function_eval
    elif stage == "rwd":
        preprocess_function = preprocess_function_train_pair

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
        print_dataset_example(dataset[0])
    elif stage == "rwd":
        print_pairwise_dataset_example(dataset[0])

    return dataset
