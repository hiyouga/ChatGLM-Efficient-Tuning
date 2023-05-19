#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_sft.py \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --dev_ratio 0.01 \
    --load_best_model_at_end \
    --plot_loss \
    --fp16
