#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python finetune_chatglm.py \
    --do_train \
    --overwrite_cache \
    --output_dir output \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 50 \
    --save_steps 1000 \
