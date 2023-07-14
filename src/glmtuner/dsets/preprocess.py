from typing import Literal
from transformers import Seq2SeqTrainingArguments
from transformers.tokenization_utils import PreTrainedTokenizer

from datasets import Dataset

from glmtuner.extras.constants import IGNORE_INDEX
from glmtuner.hparams import DataArguments


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
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
