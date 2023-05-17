#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../src/train_sft.py \
    --do_eval \
    --dataset alpaca_gpt4_zh \
    --dataset_dir ../data \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_eval_result \
    --overwrite_cache \
    --per_device_eval_batch_size 8 \
    --max_samples 50 \
    --predict_with_generate
