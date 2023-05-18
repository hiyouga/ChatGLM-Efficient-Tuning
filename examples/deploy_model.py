# coding=utf-8

import sys
sys.path.append("../src")
import torch
from src import ModelArguments, auto_configure_device_map, load_pretrained

if __name__ == "__main__":
    model_args = ModelArguments(checkpoint_dir="path_to_lora_checkpoint")
    model, tokenizer = load_pretrained(model_args)
    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(torch.cuda.device_count())
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()
    model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
