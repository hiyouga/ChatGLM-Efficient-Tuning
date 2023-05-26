import datetime
import os
import json
import time

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from transformers import TrainerCallback
from transformers.trainer import PredictionOutput
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.tokenization_utils import PreTrainedTokenizer

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .peft_trainer import PeftTrainer

from .other import (
    get_logger,
    IGNORE_INDEX,
    PREDICTION_FILE_NAME
)

logger = get_logger(__name__)


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
        # Replace IGNORE_INDEX in the labels with pad_token_id as we cannot decode them if ignore_pad_token_for_loss=True.
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for pred, label in zip(preds, labels):
            hypothesis = list(jieba.cut(self.tokenizer.decode(pred, skip_special_tokens=True)))
            reference = list(jieba.cut(self.tokenizer.decode(label, skip_special_tokens=True)))

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


class Seq2SeqTrainerForChatGLM(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

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

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
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

        generated_tokens = self.model.generate(**inputs, **gen_kwargs)
        generated_tokens = generated_tokens[:, inputs["input_ids"].size(-1):]  # important for ChatGLM

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

        loss = None  # we cannot compute loss while generation

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
        assert self.args.predict_with_generate, "Please enable `predict_with_generate` for saving model predictions."
        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions,
                         self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids,
                          self.tokenizer.pad_token_id)

        preds = [tokenizer.decode(pred, skip_special_tokens=True).strip() for pred in preds]
        labels = [tokenizer.decode(label, skip_special_tokens=True).strip() for label in labels]

        output_prediction_file = os.path.join(self.args.output_dir, PREDICTION_FILE_NAME)
        logger.info(f"Saving prediction results to {output_prediction_file}")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(preds, labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))


class MyCallback(TrainerCallback):
    r"""
    TrainerCallback Includes the state function during training, for more details refer to the TrainerCallback class.
    The on_log function primarily collects process parameters during training, such as loss rate, learning rate,
    and training epochs, as well as progress parameters like the current percentage progress and estimated remaining
    time. Every time a log is triggered, a new record is appended to the file "messages.log" for training
    visualization purposes.
    """

    def __init__(self):
        self.start_time = time.time()
        self.single_turn_time = time.time()
        self.step_times = []

    def on_log(self, args, state, control, **kwargs):
        percentage = state.log_history[-1].get('step') / state.max_steps * 100
        elapsed_time = time.time() - self.start_time
        elapsed_time_formatted = str(datetime.timedelta(seconds=int(elapsed_time)))
        self.step_times.append(time.time() - self.single_turn_time)
        self.single_turn_time = time.time()

        if state.log_history[-1].get('step') > 0:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            remaining_steps = state.max_steps - state.log_history[-1].get('step')
            remaining_time_seconds = avg_step_time * remaining_steps
            remaining_time = datetime.timedelta(seconds=int(remaining_time_seconds))
            remaining_time_formatted = str(remaining_time)
        else:
            remaining_time_formatted = "Unknown"

        # Dictionary object for returning, encapsulating the fields required for display in the frontend. Please make
        # any additions or modifications here for future changes.
        training_dict = {
            'current_step': state.log_history[-1].get('step') if state.log_history else {},
            'total_steps': state.max_steps,
            'loss': state.log_history[-1].get('loss') if state.log_history else {},
            'learning_rate': state.log_history[-1].get('learning_rate') if state.log_history else {},
            'epoch': state.log_history[-1].get('epoch') if state.log_history else {},
            'percentage': percentage,
            'elapsed_time': elapsed_time_formatted,
            'remaining_time': remaining_time_formatted
        }

        with open('messages.log', 'a') as f:
            f.write(json.dumps(training_dict) + '\n')
