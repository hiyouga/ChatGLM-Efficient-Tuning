import os
import sys
import json
import torch
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.trainer import PredictionOutput, TRAINING_ARGS_NAME
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.tokenization_utils import PreTrainedTokenizer

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .config import FinetuningArguments

from .other import (
    save_trainable_params,
    IGNORE_INDEX,
    FINETUNING_ARGS_NAME,
    PREDICTION_FILE_NAME
)


logger = logging.getLogger(__name__) # setup logging
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# Note: The ChatGLM tokenizer assigns False on token to be attended in attention mask. In general settings, it should be True.
# Refer to: https://huggingface.co/THUDM/chatglm-6b/blob/6650ae3a53c28fc176d06762ca80b05d5ab3792b/tokenization_chatglm.py#L401
class Seq2SeqDataCollatorForChatGLM(DataCollatorForSeq2Seq):
    r"""
    Data collator for ChatGLM. It is capable of dynamically padding for batched data.

    Inspired by: https://github.com/tatsu-lab/stanford_alpaca/blob/65512697dc67779a6e53c267488aba0ec4d7c02a/train.py#L156
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            ignore_pad_token_for_loss: bool,
            inference_mode: bool = False
    ):
        label_pad_token_id = IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id
        super().__init__(tokenizer, model=model, label_pad_token_id=label_pad_token_id, padding=True)
        self.label_pad_token_id = label_pad_token_id
        self.inference_mode = inference_mode

    def __call__(self, features: Sequence[Dict[str, Sequence]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        ChatGLM is able to generate attentions masks and position ids by itself.
        """
        if self.inference_mode: # evaluation set adopts left-padding while training set adopts right-padding
            return super().__call__(features)
        input_ids, labels = [[torch.tensor(feature[key]) for feature in features] for key in ("input_ids", "labels")]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)
        features = {"input_ids": input_ids, "labels": labels}
        return features


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqTrainerForChatGLM.

    Borrowed from: https://github.com/THUDM/ChatGLM-6B/blob/0c2806fea82683349194e21996dd6b3acc3c265b/ptuning/main.py#L307
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
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

            if len(" ".join(hypothesis).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}


class Seq2SeqTrainerForChatGLM(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(self, finetuning_args: FinetuningArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoints.

        This function will only be executed at the process zero.

        Override to inject custom behavior.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model_to_save = unwrap_model(self.model)

        if hasattr(self.model, "peft_config"): # peft methods
            model_to_save.save_pretrained(output_dir) # save lora weights
        else: # non-peft methods
            save_trainable_params(output_dir, model_to_save)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        torch.save(self.finetuning_args, os.path.join(output_dir, FINETUNING_ARGS_NAME))

    def prediction_step(
            self,
            model: torch.nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Performs an evaluation step on `model` using `inputs` for ChatGLM.

        Now it only supports single GPU (without Accelerate).

        Override to inject custom behavior. It is not directly used by external scripts.
        """
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
        # Inspired by: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L273
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

    def save_predictions(
            self,
            predict_results: PredictionOutput,
            tokenizer: PreTrainedTokenizer
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return
        if not self.args.predict_with_generate:
            raise ValueError("Please enable `predict_with_generate` for saving model predictions.")

        predictions = tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True)
        predictions = [pred.strip() for pred in predictions]
        labels = tokenizer.batch_decode(predict_results.label_ids, skip_special_tokens=True)
        labels = [label.strip() for label in labels]

        output_prediction_file = os.path.join(self.args.output_dir, PREDICTION_FILE_NAME)
        logger.info(f"Saving prediction results to {output_prediction_file}")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res = []
            for pred, label in zip(predictions, labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
