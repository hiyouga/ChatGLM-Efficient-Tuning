#!/usr/bin/env python
# coding=utf-8
# Implement several parameter-efficient fine-tuning method for ChatGLM.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py


import os
import torch
import logging
from utils import (
    prepare_args,
    prepare_data,
    prepare_model,
    preprocess_data,
    save_trainable_params,
    DataCollatorForChatGLM,
    ComputeMetrics,
    TrainerForChatGLM
)


logger = logging.getLogger(__name__)


def main():
    # Prepare model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args()
    dataset = prepare_data(model_args, data_args, training_args)
    tokenizer, model = prepare_model(model_args, finetuning_args)
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args)
    data_collator = DataCollatorForChatGLM(tokenizer=tokenizer, data_args=data_args)
    # Override the decoding parameters of Trainer
    training_args.generation_max_length = training_args.generation_max_length if \
                training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.num_beams if \
                data_args.num_beams is not None else training_args.generation_num_beams
    # Initialize our Trainer
    trainer = TrainerForChatGLM(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=ComputeMetrics(tokenizer, data_args) if training_args.predict_with_generate else None
    )
    # Training
    if training_args.do_train:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train()
        if finetuning_args.finetuning_type == "p_tuning":
            save_trainable_params(training_args.output_dir, model)
        elif finetuning_args.finetuning_type == "lora":
            model.save_pretrained(training_args.output_dir)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
        torch.save(finetuning_args, os.path.join(training_args.output_dir, "finetuning_args.bin"))
    # Evaluation
    if training_args.do_eval:
        model = model.half().cuda()
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=512, temperature=0.95)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    main()
