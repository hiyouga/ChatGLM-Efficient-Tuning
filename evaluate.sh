#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python finetune_chatglm.py \
    --do_eval \
    --dataset alpaca_gpt4_zh \
    --output_dir eval \
    --overwrite_cache \
    --per_device_eval_batch_size 8 \
    --max_eval_samples 20 \
    --predict_with_generate
