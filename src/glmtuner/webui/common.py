import json
import os
from typing import Dict, List, Tuple

import gradio as gr
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
from transformers.trainer import WEIGHTS_NAME, WEIGHTS_INDEX_NAME

CACHE_DIR = "cache"
DATA_DIR = "data"
SAVE_DIR = "saves"
USER_CONFIG = "user.config"


def get_config_path():
    return os.path.join(CACHE_DIR, USER_CONFIG)


def load_config() -> Dict[str, str]:
    if not os.path.exists(get_config_path()):
        return {}

    with open(get_config_path(), "r", encoding="utf-8") as f:
        try:
            user_config = json.load(f)
            return user_config
        except:
            return {}


def save_config(model_name: str, model_path: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    user_config = dict(model_name=model_name, model_path=model_path)
    with open(get_config_path(), "w", encoding="utf-8") as f:
        json.dump(user_config, f, ensure_ascii=False)


def get_save_dir(model_name: str) -> str:
    return os.path.join(SAVE_DIR, os.path.split(model_name)[-1])


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
