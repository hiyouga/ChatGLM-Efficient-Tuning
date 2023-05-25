#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_ppo.py \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --dataset_dir ../data \
    --finetuning_type lora \
    --checkpoint_dir path_to_sft_checkpoint \
    --reward_model path_to_rm_checkpoint \
    --output_dir path_to_ppo_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16
