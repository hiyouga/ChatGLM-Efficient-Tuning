import os
import sys
import jieba
import hashlib
import logging
import numpy as np
import torch
import datasets
from datasets import Dataset, concatenate_datasets, load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from peft import PeftModel, TaskType, get_peft_config, get_peft_model
from peft.peft_model import WEIGHTS_NAME
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from arguments import ModelArguments, DataTrainingArguments, FinetuningArguments


IGNORE_INDEX = -100
FINETUNING_ARGS_NAME = "finetuning_args.bin"


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


def load_trainable_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> None:
    model_state_dict = torch.load(os.path.join(checkpoint_dir, WEIGHTS_NAME))
    model.load_state_dict(model_state_dict, strict=False) # skip missing keys


# This function includes: 1. cast the laternorm in fp32 2. make output embedding layer require grads 3. upcast the lm_head to fp32
# Inspired by: https://github.com/huggingface/peft/blob/c0209c35abbf88c63aa267800d98a8e212ed0a42/src/peft/utils/other.py#L35
def prepare_model_for_training(
        model: PreTrainedModel,
        output_embedding_layer_name: Optional[str] = "lm_head",
        use_gradient_checkpointing: Optional[bool] = True,
        layer_norm_names: List[str] = ["layer_norm"]
) -> PreTrainedModel:
    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off in training for gradient checkpointing

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model


def load_pretrained(
        model_args: ModelArguments,
        finetuning_args: Optional[FinetuningArguments]=None,
        is_trainable: Optional[bool]=False
) -> Tuple[transformers.modeling_utils.PreTrainedModel, transformers.tokenization_utils.PreTrainedTokenizer]:
    # Load pretrained model and tokenizer
    if (not is_trainable) and (model_args.checkpoint_dir is None):
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    if model_args.checkpoint_dir is not None: # load fine-tuned model from checkpoint
        if not os.path.isfile(os.path.join(model_args.checkpoint_dir, FINETUNING_ARGS_NAME)):
            raise ValueError("The fine-tuning arguments are not found in the provided dictionary.")
        logger.info("Load fine-tuned model from checkpoint: {}".format(model_args.checkpoint_dir))
        finetuning_args = torch.load(os.path.join(model_args.checkpoint_dir, FINETUNING_ARGS_NAME))

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

    if finetuning_args.finetuning_type == "p_tuning":
        # use the built-in p-tuning method in ChatGLM, we cannot use peft since the attention mask is unusual >_<
        config.pre_seq_len = finetuning_args.pre_seq_len # enable this will fix other parameters automatically
        config.prefix_projection = finetuning_args.prefix_projection

    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, **config_kwargs)

    if model_args.quantization_bit is not None:
        logger.info("Quantized to {} bit".format(model_args.quantization_bit))
        model = model.quantize(model_args.quantization_bit)

    model = prepare_model_for_training(model) if is_trainable else model

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none when training.")

    if finetuning_args.finetuning_type == "freeze":
        logger.info("Fine-tuning method: Freeze")
        trainable_layers = ["layers.{:d}.mlp".format(27-k) for k in range(finetuning_args.num_layer_trainable)]
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32) # we cannot train model in half (fp16) precision

    if finetuning_args.finetuning_type == "p_tuning":
        logger.info("Fine-tuning method: P-Tuning V2")
        model.transformer.prefix_encoder.float() # we cannot train model in half (fp16) precision

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        if model_args.checkpoint_dir is not None:
            model = PeftModel.from_pretrained(model, model_args.checkpoint_dir, is_trainable=is_trainable)
            model = model if is_trainable else model.merge_and_unload() # merge LoRA weights to evaluate the model
        else:
            peft_config = {
                "peft_type": "LORA",
                "task_type": TaskType.CAUSAL_LM,
                "inference_mode": False,
                "r": finetuning_args.lora_rank,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
                "target_modules": ["query_key_value"] # query_key_value or dense
            }
            peft_config = get_peft_config(peft_config)
            model = get_peft_model(model, peft_config)
    else: # Freeze and P-Tuning
        if model_args.checkpoint_dir is not None:
            load_trainable_params(model, model_args.checkpoint_dir)

    print_trainable_params(model)

    return model, tokenizer


def prepare_args() -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments]:
    # Load arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Provide arguments with a json file.
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    if training_args.do_train and training_args.do_eval:
        raise ValueError("We don't support training and evaluation simultaneously.")
    training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim

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
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args


def prepare_data(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: Seq2SeqTrainingArguments) -> Dataset:
    # Load and verify dataset
    def checksum(file_path, hash):
        with open(file_path, "rb") as datafile:
            binary_data = datafile.read()
        sha1 = hashlib.sha1(binary_data).hexdigest()
        if sha1 != hash:
            raise ValueError("Checksum failed for {}.".format(file_path))

    max_samples = data_args.max_train_samples if training_args.do_train else data_args.max_eval_samples
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
            data_file = os.path.join(data_args.dataset_dir, dataset_info.file_name)
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
        dataset = raw_datasets["train"] # always use the training set

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
                if getattr(dataset_info, column) is not None:
                    dataset = dataset.rename_column(getattr(dataset_info, column), column_name)
                else:
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
        training_args: Seq2SeqTrainingArguments
) -> Dataset:
    # Preprocess the datasets
    column_names = list(dataset.column_names)
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    def format_example(examples):
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
                source_ids = source_ids[:data_args.max_source_length-1]
            if len(target_ids) > data_args.max_target_length - 2: # bos and eos tokens
                target_ids = target_ids[:data_args.max_target_length-2]
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

    def print_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"])))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(tokenizer.decode(example["labels"])))

    preprocess_function = preprocess_function_train if training_args.do_train else preprocess_function_eval
    # we don't provide `do_train` and `do_eval` arguments simultaneously
    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )
    print_dataset_example(dataset[0])

    return dataset


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


"""
Note: The ChatGLM tokenizer assigns False on token to be attended in attention mask. In general settings, it should be True.
Refer to: https://huggingface.co/THUDM/chatglm-6b/blob/6650ae3a53c28fc176d06762ca80b05d5ab3792b/tokenization_chatglm.py#L401
Inspired by: https://github.com/tatsu-lab/stanford_alpaca/blob/aa65c492bb788e144712daab42bc5d11c2761591/train.py#L166
"""
class DataCollatorForChatGLM(DataCollatorForSeq2Seq): # dynamically padding for training set

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            ignore_pad_token_for_loss: bool
    ):
        label_pad_token_id = IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id
        super().__init__(tokenizer, model=model, label_pad_token_id=label_pad_token_id, padding=False)
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: Sequence[Dict[str, Sequence]]) -> Dict[str, torch.Tensor]:
        if "attention_mask" in features[0]: # evaluation set adopts left-padding
            return super().__call__(features)
        input_ids, labels = tuple([torch.tensor(feature[key]) for feature in features] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)
        features = {"input_ids": input_ids, "labels": labels}
        return features


"""
Borrowed from: https://github.com/THUDM/ChatGLM-6B/blob/0c2806fea82683349194e21996dd6b3acc3c265b/ptuning/main.py#L307
"""
@dataclass
class ComputeMetrics:

    tokenizer: PreTrainedTokenizer

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace IGNORE_INDEX in the labels with pad_token_id as we cannot decode them if ignore_pad_token_for_loss=True.
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}


"""
Inspired by: https://github.com/mymusise/ChatGLM-Tuning/blob/997393046a49510e6cda36962f9a399297959311/finetune.py#L52
Use Seq2SeqTrainer to compute generative metrics such as BLEU, ROUGE, and etc.
However, the evaluation seems very slow, it will be resolved in the future.
"""
class TrainerForChatGLM(Seq2SeqTrainer):

    def _save(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        if hasattr(self.model, "peft_config"): # LoRA
            self.model.save_pretrained(output_dir) # only save peft weights with the built-in method
        else:
            save_trainable_params(output_dir, self.model) # Freeze and P-Tuning
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def save_finetuning_args(self, finetuning_args: FinetuningArguments) -> None:
        torch.save(finetuning_args, os.path.join(self.args.output_dir, FINETUNING_ARGS_NAME))

    def prediction_step(
            self,
            model: torch.nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Override to inject custom bevavior.
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = gen_kwargs["num_beams"] \
                    if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = gen_kwargs["synced_gpus"] \
                    if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "position_ids" in inputs:
            gen_kwargs["position_ids"] = inputs.get("position_ids", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        gen_kwargs["input_ids"] = generation_inputs
        generated_tokens = self.model.generate(**gen_kwargs)
        generated_tokens = generated_tokens[:, generation_inputs.size()[-1]:] # important for ChatGLM

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # Inspired by: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_seq2seq.py#L273
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels
