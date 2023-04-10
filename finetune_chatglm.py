#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
# This code is largely borrowed from the following repositories:
# [1] https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
# [2] https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
# [3] https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
# [4] https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/Chatglm6b_ModelParallel_ptuning/main.py


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
    TrainerForChatGLM
)


logger = logging.getLogger(__name__)


def main():

    model_args, data_args, training_args, finetuning_args = prepare_args()
    raw_datasets = prepare_data(model_args, data_args)
    tokenizer, model = prepare_model(model_args, finetuning_args)
    dataset = preprocess_data(raw_datasets, tokenizer, data_args, training_args)
    data_collator = DataCollatorForChatGLM(tokenizer=tokenizer, data_args=data_args)
    trainer = TrainerForChatGLM(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    # Training
    if training_args.do_train:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        trainer.train()
        if finetuning_args.finetuning_type == "p_tuning":
            save_trainable_params(training_args.output_dir, model)
        elif finetuning_args.finetuning_type == "lora":
            model.save_pretrained(training_args.output_dir)
        torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))


if __name__ == '__main__':
    main()
