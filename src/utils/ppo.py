import os
import json
import math
import torch
from tqdm import tqdm
from typing import Callable, Dict, List, Literal, Optional, Tuple

from transformers import Seq2SeqTrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME, TRAINER_STATE_NAME
from transformers.modeling_utils import PreTrainedModel

from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.trainer.ppo_trainer import PPODecorators, logprobs_from_logits

from .config import FinetuningArguments

from .other import (
    AverageMeter,
    get_logger,
    get_state_dict,
    get_logits_processor,
    FINETUNING_ARGS_NAME,
    VALUE_HEAD_FILE_NAME
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


class PPOTrainerForChatGLM(PPOTrainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(self, training_args: Seq2SeqTrainingArguments, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(**kwargs)
        self.data_collator = self.accelerator.prepare(kwargs["data_collator"])
        self.state = {"log_history": []}
        self.training_args = training_args
        self.finetuning_args = finetuning_args

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
        unwrapped_model: PreTrainedModel = self.accelerator.unwrap_model(self.model)

        dataiter = iter(self.dataloader)
        steps_trained = 0
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()

        for step in tqdm(range(max_steps), disable=not self.is_world_process_zero()):

            for _ in range(self.config.gradient_accumulation_steps):

                batch = next(dataiter)
                steps_trained += 1

                unwrapped_model.gradient_checkpointing_disable()
                unwrapped_model.config.use_cache = True

                # Get response from ChatGLM
                query_tensors: torch.Tensor = batch["input_ids"]
                response_tensors = self.generate(batch, length_sampler=output_length_sampler, return_prompt=False, **gen_kwargs)

                queries: List[torch.Tensor] = []
                responses: List[torch.Tensor] = []
                for i in range(len(query_tensors)):
                    query_length = (query_tensors[i] != self.tokenizer.pad_token_id).nonzero()[0]
                    response_length = (response_tensors[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1
                    queries.append(query_tensors[i, query_length:]) # remove padding from left
                    if response_length < 2: # make response have at least 2 tokens
                        responses.append(response_tensors.new_empty(2).fill_(self.tokenizer.eos_token_id))
                    else:
                        responses.append(response_tensors[i, :response_length]) # remove padding from right

                # Compute rewards
                replace_model(unwrapped_model, target="reward")
                _, _, values = self.model(**self.prepare_model_inputs(queries, responses))
                rewards = [reward for reward in values[-1]]
                replace_model(unwrapped_model, target="default") # make sure the model is default at the end

                # Run PPO step
                unwrapped_model.gradient_checkpointing_enable()
                unwrapped_model.config.use_cache = False

                stats = self.step(queries, responses, rewards)

                loss_meter.update(stats["ppo/loss/total"])
                reward_meter.update(torch.tensor(rewards).sum().item(), n=len(rewards))

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

    @torch.no_grad()
    def generate(
            self,
            inputs: Dict[str, torch.Tensor],
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

        response = unwrapped_model.generate(**inputs, **generation_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # Inspired by: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L273
        if unwrapped_model.pretrained_model.generation_config._from_model_config:
            unwrapped_model.pretrained_model.generation_config._from_model_config = False

        self.model, _ = cast_layernorm_dtype(self.model, layer_norm_params)

        if not return_prompt and not self.is_encoder_decoder:
            return response[:, inputs["input_ids"].size(1):]
        return response

    def prepare_model_inputs(self, queries: List[torch.Tensor], responses: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        input_data = self.data_collator([{"input_ids": ids} for ids in input_ids])
        input_data = {k: v.to(self.current_device) for k, v in input_data.items() if v is not None}
        input_data.pop("labels", None)  # we don't want to compute LM losses
        return input_data

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
        bs = len(model_inputs["input_ids"])
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {k: v[i * fbs : (i + 1) * fbs] for k, v in model_inputs.items()}
            input_ids: torch.Tensor = input_kwargs["input_ids"] # left-padded sequences
            if self.is_distributed: # re-generate them to adapt padded inputs
                input_kwargs["attention_mask"] = self.data_collator.get_attention_masks(input_ids, device=self.current_device)
                input_kwargs["position_ids"] = self.data_collator.get_position_ids(input_ids, device=self.current_device)
            logits, _, values = model(**input_kwargs)
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

            values = values.transpose(0, 1)
            masks = torch.zeros_like(input_ids)

            for j in range(fbs):
                start = (input_ids[j] == self.tokenizer.bos_token_id).nonzero()[0].item()
                masks[j][start:] = 1
                if len(masks[j][start:]) < 2:
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
        Saves trainable parameters as model checkpoint.

        Subclass and override to inject custom behavior.
        """
        self.accelerator.wait_for_everyone() # must be executed before is_world_process_zero()
        if not self.is_world_process_zero():
            return

        output_dir = output_dir if output_dir is not None else self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = self.accelerator.unwrap_model(self.model)
        model.pretrained_model.save_pretrained(output_dir, state_dict=get_state_dict(model.pretrained_model)) # lora weights
        torch.save(get_state_dict(model.v_head), os.path.join(output_dir, VALUE_HEAD_FILE_NAME)) # valuehead weights
        torch.save(self.training_args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        torch.save(self.finetuning_args, os.path.join(output_dir, FINETUNING_ARGS_NAME))
