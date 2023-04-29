# coding=utf-8
# Implements parameter-efficient ppo training of fine-tuned ChatGLM.
# This code is inspired by:
# https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py

from tqdm import tqdm

import torch
from torch.optim import AdamW

from trl import PPOConfig
from trl.core import LengthSampler

from utils import (
    prepare_args,
    prepare_data,
    load_pretrained,
    preprocess_data,
    PPODataCollatorForChatGLM,
    PPOTrainerForChatGLM,
    compute_rewards
)


def main():

    # prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args()
    dataset = prepare_data(model_args, data_args)
    model, tokenizer = load_pretrained(model_args, training_args, finetuning_args, training_args.do_train, stage="ppo")
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="ppo")
    data_collator = PPODataCollatorForChatGLM(
        tokenizer=tokenizer,
        min_input_length=data_args.max_source_length, # avoid truncating input sequences
        max_input_length=data_args.max_source_length,
        inference_mode=(not training_args.do_train)
    )

    ppo_config = PPOConfig(
        model_name=model_args.model_name_or_path,
        learning_rate=training_args.learning_rate,
        mini_batch_size=max(training_args.per_device_train_batch_size // 4, 1),
        batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=int(training_args.num_train_epochs),
        max_grad_norm=training_args.max_grad_norm
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=ppo_config.learning_rate)

    # Initialize our Trainer
    ppo_trainer = PPOTrainerForChatGLM(
        finetuning_args=finetuning_args,
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator,
        optimizer=optimizer
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    output_length_sampler = LengthSampler(data_args.max_target_length // 2, data_args.max_target_length)

    step = 0
    for batch in tqdm(ppo_trainer.dataloader):
        queries = batch["input_ids"] # left-padded sequences

        model.gradient_checkpointing_disable()
        model.config.use_cache = True

        # Get response from ChatGLM
        responses_with_queries = ppo_trainer.generate(queries, length_sampler=output_length_sampler, **gen_kwargs)
        responses = responses_with_queries[:, queries.size(1):] # right-padded sequences
        batch["response"] = tokenizer.batch_decode(responses, skip_special_tokens=True)

        for i in range(responses_with_queries.size(0)): # change to right-padding
            start = (responses_with_queries[i] != tokenizer.pad_token_id).nonzero()[0].item()
            responses_with_queries[i] = torch.cat((responses_with_queries[i][start:], responses_with_queries[i][:start]))

        # Compute rewards
        rewards = compute_rewards(responses_with_queries, model, tokenizer)

        # Run PPO step
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        split_into_list = lambda x: [x[i] for i in range(x.size(0))]
        stats = ppo_trainer.step(*map(split_into_list, [queries, responses, rewards]))
        ppo_trainer.log_stats(stats, batch, rewards)
        if step % training_args.logging_steps == 0:
            print("{{'loss': {:.4f}, 'learning_rate': {:}}}".format(stats["ppo/loss/total"], stats["ppo/learning_rate"]))
        step += 1

    ppo_trainer.save_model(training_args.output_dir)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
