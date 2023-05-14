#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_rm.py \
    --do_train \
    --dataset comparison_gpt4_zh \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir path_to_rm_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --fp16
