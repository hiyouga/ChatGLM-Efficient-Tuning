#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/finetune.py \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir output_finetune \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --max_train_samples 10000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --fp16
