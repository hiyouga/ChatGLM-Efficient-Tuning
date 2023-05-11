#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/finetune.py \
    --do_train \
    --model_name_or_path path_to_chatglm_model \
    --dataset alpaca_gpt4_zh \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir path_to_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --quantization_bit 8 \
    --fp16
