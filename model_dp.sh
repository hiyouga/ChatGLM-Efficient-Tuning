#!/bin/bash

# params
gpu_id='3'
port='6103'
model='/rec_reason_4bit_v1/hf'

# default
model_base_dir="/data/jupyterlab/gzx/LocalModelHub"

CUDA_VISIBLE_DEVICES=${gpu_id} python src/api_fine_tuned.py \
	--app_port ${port} \
	--model_dir ${model_base_dir}${model}
