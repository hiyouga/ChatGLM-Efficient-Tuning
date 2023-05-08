import os
import sys
import torch
import logging
from typing import Dict, Optional, Sequence

from transformers import Trainer, DataCollatorWithPadding
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.modeling_utils import unwrap_model
from transformers.tokenization_utils import PreTrainedTokenizer

from .config import FinetuningArguments

from .other import (
    save_trainable_params,
    save_valuehead_params,
    FINETUNING_ARGS_NAME
)


logger = logging.getLogger(__name__) # setup logging
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class PairwiseDataCollatorForChatGLM(DataCollatorWithPadding):
    r"""
    Data collator for ChatGLM. It is capable of dynamically padding for batched data.

    Inspired by: https://github.com/tatsu-lab/stanford_alpaca/blob/65512697dc67779a6e53c267488aba0ec4d7c02a/train.py#L156
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            inference_mode: bool = False
    ):
        super().__init__(tokenizer, padding=True)
        self.inference_mode = inference_mode

    def __call__(self, features: Sequence[Dict[str, Sequence]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch. We adopt right-padding for pairwise data.

        We generate 2 * n examples where the first n examples represents chosen examples and
        the last n examples represents rejected examples.

        ChatGLM is able to generate attentions masks and position ids by itself.
        """
        if self.inference_mode:
            raise NotImplementedError
        accept_ids, reject_ids = [[torch.tensor(feature[key]) for feature in features] for key in ("accept_ids", "reject_ids")]
        input_ids = accept_ids + reject_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        features = {"input_ids": input_ids}
        return features

class PairwiseTrainerForChatGLM(Trainer):
    r"""
    Inherits Trainer to compute pairwise loss.
    """

    def __init__(self, finetuning_args: FinetuningArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetuning_args = finetuning_args

    def compute_loss(self, model, inputs, return_outputs=False):
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.

        We use score on the EOS token to represent reward of the whole sentence.
        """
        batch_size = inputs["input_ids"].size(0) // 2
        _, _, values = model(input_ids=inputs["input_ids"])
        rewards = values.transpose(0, 1)[(inputs["input_ids"] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)]
        r_accept, r_reject = rewards.split(batch_size, dim=0)
        loss = -torch.log(torch.sigmoid(r_accept - r_reject)).mean()
        if return_outputs:
            return loss, {"r_accept": r_accept, "r_reject": r_reject}
        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoints. Use `self.model.pretrained_model` to refer to the backbone model.

        This function will only be executed at the process zero.

        Override to inject custom behavior.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model_to_save = unwrap_model(self.model)

        if hasattr(model_to_save.pretrained_model, "peft_config"): # peft methods
            model_to_save.pretrained_model.save_pretrained(output_dir) # save lora weights
        else: # non-peft methods
            save_trainable_params(output_dir, model_to_save.pretrained_model)

        if hasattr(model_to_save, "v_head"):
            save_valuehead_params(output_dir, model_to_save.v_head) # save valuehead weights

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        torch.save(self.finetuning_args, os.path.join(output_dir, FINETUNING_ARGS_NAME))
