#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python finetune_chatglm.py \
    --do_train \
    --dataset guanaco \
    --output_dir output \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 100 \
    --max_train_samples 10000 \
    --learning_rate 5e-4 \
    --num_train_epochs 1.0 \
    --finetuning_type lora \
    --fp16
