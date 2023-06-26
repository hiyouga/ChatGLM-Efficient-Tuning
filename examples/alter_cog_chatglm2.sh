#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_sft.py \
    --model_name_or_path path_to_chatglm2_6b \
    --use_v2 \
    --do_train \
    --dataset self_cognition \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --learning_rate 1e-3 \
    --num_train_epochs 20.0 \
    --fp16

CUDA_VISIBLE_DEVICES=0 python ../src/train_sft.py \
    --model_name_or_path path_to_chatglm2_6b \
    --use_v2 \
    --do_predict \
    --dataset self_cognition \
    --dataset_dir ../data \
    --checkpoint_dir path_to_sft_checkpoint \
    --output_dir path_to_predictions \
    --overwrite_cache \
    --per_device_eval_batch_size 8 \
    --predict_with_generate
