import codecs
import json
import os
from typing import List, Tuple

import gradio as gr
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
from transformers.trainer import WEIGHTS_NAME, WEIGHTS_INDEX_NAME

CACHE_DIR = "cache"  # to save models
DATA_DIR = "data"
SAVE_DIR = "saves"
TEMP_USE_CONFIG = "tmp.use.config"


def get_temp_use_config_path():
    return os.path.join(SAVE_DIR, TEMP_USE_CONFIG)


def load_temp_use_config():
    if not os.path.exists(get_temp_use_config_path()):
        return {}
    with codecs.open(get_temp_use_config_path()) as f:
        try:
            user_config = json.load(f)
            return user_config
        except Exception as e:
            return {}


def save_temp_use_config(user_config: dict):
    with codecs.open(get_temp_use_config_path(), "w", encoding="utf-8") as f:
        json.dump(f, user_config, ensure_ascii=False)


def save_model_config(model_name: str, model_path: str):
    with codecs.open(get_temp_use_config_path(), "w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, "model_path": model_path}, f, ensure_ascii=False)


def get_save_dir(model_name: str) -> str:
    return os.path.join(SAVE_DIR, model_name.split("/")[-1])


def add_model(model_list: list, model_name: str, model_path: str) -> Tuple[list, str, str]:
    model_list = model_list + [[model_name, model_path]]
    return model_list, "", ""


def del_model(model_list: list, model_name: str) -> list:
    model_list = [[name, path] for name, path in model_list if name != model_name]
    return model_list


def list_models(model_list: list) -> dict:
    return gr.update(value="", choices=[name for name, _ in model_list])


def list_checkpoints(model_name: str) -> dict:
    checkpoints = []
    save_dir = get_save_dir(model_name)
    if save_dir and os.path.isdir(save_dir):
        for checkpoint in os.listdir(save_dir):
            if (
                    os.path.isdir(os.path.join(save_dir, checkpoint))
                    and any([
                os.path.isfile(os.path.join(save_dir, checkpoint, name))
                for name in (WEIGHTS_NAME, WEIGHTS_INDEX_NAME, PEFT_WEIGHTS_NAME)
            ])
            ):
                checkpoints.append(checkpoint)
    return gr.update(value=[], choices=checkpoints)


def list_datasets() -> List[str]:
    with open(os.path.join(DATA_DIR, "dataset_info.json"), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
        return list(dataset_info.keys())
