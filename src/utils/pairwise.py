import os
import torch
from typing import Dict, Optional, Sequence, Union

from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME

from .config import FinetuningArguments

from .data_collator import DataCollatorForChatGLM

from .other import (
    get_logger,
    save_trainable_params,
    FINETUNING_ARGS_NAME
)


logger = get_logger(__name__)


class PairwiseDataCollatorForChatGLM(DataCollatorForChatGLM):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Union[torch.Tensor, Sequence[int]]]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        features = [{"input_ids": feature[key]} for feature in features for key in ("accept_ids", "reject_ids")]
        return super().__call__(features)


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
        _, _, values = model(**inputs)
        r_accept, r_reject = values[-1].split(batch_size, dim=0)
        loss = -torch.log(torch.sigmoid(r_accept - r_reject)).mean()
        if return_outputs:
            return loss, {"r_accept": r_accept, "r_reject": r_reject}
        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoints.

        This function will only be executed at the process zero.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        save_trainable_params(output_dir, self.model)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        torch.save(self.finetuning_args, os.path.join(output_dir, FINETUNING_ARGS_NAME))
