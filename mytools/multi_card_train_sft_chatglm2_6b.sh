#!/bin/bash

# date
date=`date +%Y%m%d%H`

# params
model="rec_reason_v1_chatglm26b"
sft_data="rec_reason_train"
output_dir="./model/"${model}"_ckp"
log_dir="./log/"${model}"_"${date}

# default
model_base_dir="/home/apps/gzx/LocalModelHub"
chatglm2_6b=${model_base_dir}"/chatglm2_6b/hf"

set -x
accelerate launch ./src/train_sft.py \
    --model_name_or_path ${chatglm2_6b} \
    --use_v2 \
    --do_train \
    --dataset  ${sft_data}\
    --finetuning_type lora \
    --output_dir ${output_dir} \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --learning_rate 1e-3 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --ddp_find_unused_parameters False \
    --num_train_epochs 3.0 \
    --fp16 &>> ${log_dir}
