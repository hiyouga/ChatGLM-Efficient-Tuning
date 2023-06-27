#!/bin/bash

# params
output_model="rec_reason_v1"
checkpoint="./model/rec_reason_sft_checkpoint"

# default
model_base_dir="/data/jupyterlab/gzx/LocalModelHub"
chatglm_6b=${model_base_dir}"/chatglm_6b/hf"
output_dir=${model_base_dir}"/${output_model}/hf"

python src/export_model.py \
	--model_name_or_path ${chatglm_6b} \
        --checkpoint_dir ${checkpoint} \
	--output_dir ${output_dir}
