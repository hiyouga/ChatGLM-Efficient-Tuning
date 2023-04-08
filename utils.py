import os
import sys
import hashlib
import logging
import logging
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)
from typing import Type, Dict, Sequence, Optional
from dataclasses import dataclass
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from arguments import ModelArguments, DataTrainingArguments, FinetuningArguments


IGNORE_INDEX = -100
logger = logging.get_logger(__name__)


def prepare_args():
    # Load arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, FinetuningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Provide argumetns with a json file.
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args


def prepare_data(model_args, data_args):
    # Load and verify dataset
    def checksum(filepath, hash):
        with open(filepath, "rb") as datafile:
            binary_data = datafile.read()
        sha1 = hashlib.sha1(binary_data).hexdigest()
        if sha1 != hash:
            raise ValueError("Checksum failed for {}.".format(filepath))

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = os.path.join(data_args.dataset_dir, data_args.train_file)
        extension = data_args.train_file.split(".")[-1]
        checksum(data_files["train"], data_args.train_hash)

    if data_args.validation_file is not None:
        data_files["validation"] = os.path.join(data_args.dataset_dir, data_args.validation_file)
        extension = data_args.validation_file.split(".")[-1]
        checksum(data_files["validation"], data_args.validation_hash)

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None
    )
    return raw_datasets


def prepare_model(model_args, finetuning_args):
    # Load pretrained model and tokenizer
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        **config_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        **config_kwargs
    )
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, **config_kwargs)

    if model_args.quantization_bit is not None:
        print("Quantized to {} bit".format(model_args.quantization_bit))
        model = model.quantize(model_args.quantization_bit)

    if finetuning_args.finetuning_type == 'lora':
        for param in model.parameters():
            param.requires_grad_(False)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetuning_args.lora_rank,
            lora_alpha=finetuning_args.lora_alpha,
            lora_dropout=finetuning_args.lora_dropout,
            target_modules=['query_key_value'] # query_key_value or dense
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model = model.half()

    return tokenizer, model


def preprocess_data(raw_datasets, tokenizer, data_args, training_args):
    # Preprocess the datasets
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("This datasets does not contain a training set.")
        column_names = list(raw_datasets["train"].column_names)
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("This datasets does not contain a validation set.")
        column_names = list(raw_datasets["validation"].column_names)
    else:
        raise ValueError("Please specify `do_train` or `do_eval`.")
    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    query_column = data_args.query_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    def format_example(examples):
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]
                if query_column is not None and examples[query_column][i]:
                    query += examples[query_column][i]
                if history_column is not None and examples[history_column][i]:
                    prompt = ""
                    history = examples[history_column][i]
                    for i, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                else:
                    prompt = query
                prompt = prefix + prompt
                yield prompt, answer

    def preprocess_function_train(examples):
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 1: # gmask or eos token
                source_ids = source_ids[:data_args.max_source_length-1]
            if len(target_ids) > data_args.max_target_length - 2: # bos and eop tokens
                target_ids = target_ids[:data_args.max_target_length-2]
            input_ids = tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            labels = [IGNORE_INDEX] * context_length + input_ids[context_length:]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs

    def preprocess_function_eval(examples):
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt)
            target_ids = tokenizer.encode(text=answer)

            if len(source_ids) > data_args.max_source_length:
                source_ids = source_ids[:data_args.max_source_length]
            if len(target_ids) > data_args.max_target_length:
                target_ids = target_ids[:data_args.max_target_length]

            model_inputs["input_ids"].append(source_ids)
            model_inputs["labels"].append(target_ids)
        return model_inputs

    def print_dataset_example(example):
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset"
            )
        print_dataset_example(train_dataset[0])
        return train_dataset

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset"
            )
        print_dataset_example(eval_dataset[0])
        return eval_dataset


"""
Note: The ChatGLM tokenizer assigns False on token to be attended in attention mask. In general settings, it should be True.
Refer to: https://huggingface.co/THUDM/chatglm-6b/blob/6650ae3a53c28fc176d06762ca80b05d5ab3792b/tokenization_chatglm.py#L401
Inspired by: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L166
"""
@dataclass
class DataCollatorForChatGLM(DataCollatorWithPadding):
    # Use collator to pad each sequence to the longest sequence in a batch
    data_args: Type[dataclass] = None # required

    def __call__(self, features: Sequence[Dict[str, Sequence]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.tensor(feature[key]) for feature in features] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        label_pad_token_id = IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=label_pad_token_id)
        features = {"input_ids": input_ids, "labels": labels}
        return super().__call__(features)


"""
Inspired by: https://github.com/mymusise/ChatGLM-Tuning/blob/master/finetune.py#L52
"""
class TrainerForChatGLM(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        return model(**inputs).loss
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        from transformers.trainer import TRAINING_ARGS_NAME
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        state_dict = {
            k: v.cpu() for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(state_dict, os.path.join(output_dir, "lora_params.bin"))


# TODO: compute_metrics with dataclass
