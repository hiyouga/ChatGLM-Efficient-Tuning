# coding=utf-8

import sys
sys.path.append("../src")
from src import ModelArguments, load_pretrained

if __name__ == "__main__":
    model_args = ModelArguments(checkpoint_dir="path_to_lora_checkpoint")
    model, tokenizer = load_pretrained(model_args)
    model = model.get_base_model()
    model._keys_to_ignore_on_save = "lora"
    model.save_pretrained("path_to_save_model", max_shard_size="1GB")
    tokenizer.save_pretrained("path_to_save_model")
