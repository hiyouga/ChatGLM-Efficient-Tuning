import os
import json
import math
import torch
from tqdm import tqdm
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

from transformers import DataCollatorWithPadding, Seq2SeqTrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME, TRAINER_STATE_NAME
from transformers.tokenization_utils import PreTrainedTokenizer

from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.trainer.ppo_trainer import PPODecorators, logprobs_from_logits

from .config import FinetuningArguments

from .other import (
    AverageMeter,
    get_logger,
    save_trainable_params,
    save_valuehead_params,
    get_logits_processor,
    FINETUNING_ARGS_NAME
)


logger = get_logger(__name__)


def replace_model(model: AutoModelForCausalLMWithValueHead, target: Literal["default", "reward"]) -> None:
    if target == "reward": # save original head temporarily
        valuehead_state_dict = model.v_head.state_dict()

        setattr(model, "origin_head_weight", valuehead_state_dict["summary.weight"])
        setattr(model, "origin_head_bias", valuehead_state_dict["summary.bias"])

    model.pretrained_model.set_adapter(target) # set the LoRA adapter to be active
    model.v_head.load_state_dict({
        "summary.weight": getattr(model, "{}_head_weight".format(target)),
        "summary.bias": getattr(model, "{}_head_bias".format(target))
    })


@torch.no_grad()
def compute_rewards(
        input_ids: torch.Tensor, # (batch size x seq len) with format `X [gMASK] [BOS] Y [EOS] [PAD] ... [PAD]`
        model: AutoModelForCausalLMWithValueHead,
        tokenizer: PreTrainedTokenizer
) -> torch.Tensor:

    replace_model(model, target="reward")

    _, _, values = model(input_ids=input_ids)
    values = values.transpose(0, 1)

    rewards = []
    for i in range(input_ids.size(0)):
        eos_idx = (input_ids[i] == tokenizer.eos_token_id).nonzero() # Note: checking with [EOS] token is unsafe
        if len(eos_idx):
            eos_idx = eos_idx[0].item()
        else:
            eos_idx = input_ids.size(1) - 1
        rewards.append(values[i][eos_idx])
    rewards = torch.stack(rewards, dim=0)

    replace_model(model, target="default")

    return rewards


def cast_layernorm_dtype(
        model: AutoModelForCausalLMWithValueHead,
        layer_norm_names: List[str] = ["layernorm"], # for chatglm setting
        layer_norm_params: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[AutoModelForCausalLMWithValueHead, Dict[str, torch.Tensor]]:

    layer_norm_state_dict = {}

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            if layer_norm_params is not None:
                param.data = layer_norm_params[name] # restore float32 weights
            else:
                layer_norm_state_dict[name] = param.data.detach().clone() # store float32 weights for stability
                param.data = param.data.to(torch.float16)

    return model, layer_norm_state_dict


class PPODataCollatorForChatGLM(DataCollatorWithPadding):
    r"""
    Data collator for ChatGLM. It is capable of dynamically padding for batched data.
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            min_input_length: int,
            max_input_length: int,
            inference_mode: bool = False,
    ):
        super().__init__(tokenizer, padding=True)
        self.inference_mode = inference_mode

        if min_input_length < max_input_length:
            self.input_size = LengthSampler(min_input_length, max_input_length)
        else:
            self.input_size = lambda: max_input_length # always use max_input_length

    def __call__(self, features: Sequence[Dict[str, Sequence]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch. We adopt left-padding for ppo data.

        Equips with a length sampler to generate sequences with variable lengths.

        ChatGLM is able to generate attentions masks and position ids by itself.
        """
        if self.inference_mode:
            raise NotImplementedError

        input_ids = [torch.tensor(feature["input_ids"][:self.input_size()]).flip(0) for feature in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        features = {"input_ids": input_ids.flip(-1)}
        return features


class PPOTrainerForChatGLM(PPOTrainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(self, training_args: Seq2SeqTrainingArguments, finetuning_args: FinetuningArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = {"log_history": []}
        self.training_args = training_args
        self.finetuning_args = finetuning_args

    @torch.no_grad()
    def generate(
            self,
            query_tensor: torch.Tensor, # (batch size x seq len)
            length_sampler: Callable = None,
            return_prompt: bool = True,
            **generation_kwargs,
    ) -> torch.Tensor:
        r"""
        Generate response with the model given the query tensor.

        Inspired by: https://github.com/lvwerra/trl/blob/08f550674c553c36c51d1027613c29f14f3676a5/trl/trainer/ppo_trainer.py#L387
        """

        self.model, layer_norm_params = cast_layernorm_dtype(self.model)

        if length_sampler is not None:
            generation_kwargs["max_new_tokens"] = length_sampler()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        response = unwrapped_model.generate(
            input_ids=query_tensor, **generation_kwargs
        )

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # Inspired by: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L273
        if unwrapped_model.pretrained_model.generation_config._from_model_config:
            unwrapped_model.pretrained_model.generation_config._from_model_config = False

        self.model, _ = cast_layernorm_dtype(self.model, layer_norm_params)

        if not return_prompt and not self.is_encoder_decoder:
            return response[:, query_tensor.size(1):]
        return response

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        input_ids = []
        for query, response in zip(queries, responses): # query is left-padded, response is right-padded
            start = (query != self.tokenizer.pad_token_id).nonzero()[0].item()
            input_ids.append(torch.cat((query[start:], response, query[:start]))) # change to right-padding

        model_inputs =  {"input_ids": torch.stack(input_ids, dim=0).to(self.current_device)} # already padded to equal length
        model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"]) # unused indeed, avoid distributed error
        return model_inputs

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: AutoModelForCausalLMWithValueHead,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
    ):
        r"""
        Calculate model outputs in multiple batches.

        Override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}

            input_ids = input_kwargs["input_ids"]
            logits, _, values = model(input_ids=input_ids) # chatglm only needs input_ids
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

            values = values.transpose(0, 1)
            masks = torch.zeros_like(input_ids)

            for j in range(fbs):
                start = (input_ids[j] == self.tokenizer.bos_token_id).nonzero()[0].item() # always contain a [BOS] token
                end = (input_ids[j] == self.tokenizer.eos_token_id).nonzero() # Note: checking with [EOS] token is unsafe
                if len(end):
                    end = end[0].item()
                else:
                    end = masks.size(1)
                masks[j][start:end] = 1
                if end - start < 2:
                    raise ValueError("Responses are too short. Make sure they are at least 4 tokens long.")

            all_logits.append(logits)
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1],
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def ppo_train(self, max_target_length: int) -> None:

        total_train_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps * self.training_args.world_size
        len_dataloader = len(self.dataloader)
        num_steps_per_epoch = max(len_dataloader // self.config.gradient_accumulation_steps, 1)
        num_examples = len(self.dataset)
        num_train_epochs = self.training_args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * num_steps_per_epoch)

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.config.batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  Number of trainable parameters = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # Keyword arguments for `model.generate`
        gen_kwargs = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "logits_processor": get_logits_processor()
        }
        output_length_sampler = LengthSampler(max_target_length // 2, max_target_length)
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        dataiter = iter(self.dataloader)
        steps_trained = 0
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()

        for step in tqdm(range(max_steps)):

            for _ in range(self.config.gradient_accumulation_steps):

                batch = next(dataiter)
                steps_trained += 1
                queries = batch["input_ids"] # left-padded sequences

                unwrapped_model.gradient_checkpointing_disable()
                unwrapped_model.config.use_cache = True

                # Get response from ChatGLM
                responses_with_queries = self.generate(queries, length_sampler=output_length_sampler, **gen_kwargs)
                responses = responses_with_queries[:, queries.size(1):].clone().detach() # right-padded sequences (remember to clone!!!)
                # batch["response"] = tokenizer.batch_decode(responses, skip_special_tokens=True) # comment to avoid decode error

                for i in range(responses_with_queries.size(0)): # change to right-padding
                    start = (responses_with_queries[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
                    responses_with_queries[i] = torch.cat((responses_with_queries[i][start:], responses_with_queries[i][:start]))

                # Compute rewards
                rewards = compute_rewards(responses_with_queries, unwrapped_model, self.tokenizer)

                # Run PPO step
                unwrapped_model.gradient_checkpointing_enable()
                unwrapped_model.config.use_cache = False

                split_into_list = lambda x: [x[i] for i in range(x.size(0))]
                stats = self.step(*map(split_into_list, [queries, responses, rewards]))

                loss_meter.update(stats["ppo/loss/total"])
                reward_meter.update(rewards.sum().item(), n=rewards.size(0))

                if steps_trained == len_dataloader:
                    dataiter = iter(self.dataloader)
                    steps_trained = 0

            if self.is_world_process_zero() and (step+1) % self.training_args.logging_steps == 0:
                logs = {
                    "loss": round(loss_meter.avg, 4),
                    "reward": round(reward_meter.avg, 4),
                    "learning_rate": stats["ppo/learning_rate"],
                    "epoch": round(step / num_steps_per_epoch, 2)
                }
                print(logs)
                logs["step"] = step
                self.state["log_history"].append(logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step+1) % self.training_args.save_steps == 0: # save checkpoint
                self.save_model(os.path.join(self.training_args.output_dir, f"checkpoint-{step+1}"))

    def is_world_process_zero(self) -> bool:
        r"""
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        return self.training_args.process_index == 0

    def save_state(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves trainer state.
        """
        if not self.is_world_process_zero():
            return

        output_dir = output_dir if output_dir is not None else self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        json.dump(self.state, open(os.path.join(output_dir, TRAINER_STATE_NAME), "w", encoding="utf-8", newline="\n"), indent=2)

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves trainable parameters as model checkpoints. We use `self.model.pretrained_model` to refer to the backbone model.

        Override to inject custom behavior.
        """
        if not self.is_world_process_zero():
            return

        output_dir = output_dir if output_dir is not None else self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if hasattr(unwrapped_model.pretrained_model, "peft_config"): # peft methods
            unwrapped_model.pretrained_model.save_pretrained(output_dir) # save lora weights
        else: # non-peft methods
            save_trainable_params(output_dir, unwrapped_model.pretrained_model)

        if hasattr(unwrapped_model, "v_head"):
            save_valuehead_params(output_dir, unwrapped_model.v_head) # save valuehead weights

        torch.save(self.training_args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        torch.save(self.finetuning_args, os.path.join(output_dir, FINETUNING_ARGS_NAME))
