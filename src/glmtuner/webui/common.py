import json
import os
from typing import Any, Dict, List, Optional, Union

import gradio as gr
from peft.utils import WEIGHTS_NAME as PEFT_WEIGHTS_NAME
from transformers.trainer import WEIGHTS_NAME, WEIGHTS_INDEX_NAME


DEFAULT_CACHE_DIR = "cache"
DEFAULT_DATA_DIR = "data"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user.config"
DATA_CONFIG = "dataset_info.json"
METHODS = ["full", "freeze", "p_tuning", "lora"]


def get_config_path():
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)


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
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = dict(model_name=model_name, model_path=model_path)
    with open(get_config_path(), "w", encoding="utf-8") as f:
        json.dump(user_config, f, ensure_ascii=False)


def get_save_dir(model_name: str) -> str:
    return os.path.join(DEFAULT_SAVE_DIR, os.path.split(model_name)[-1])


def list_models(model_list: list) -> dict: # TODO: unused
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


def load_dataset_info(dataset_dir: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(dataset_dir, DATA_CONFIG), "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def list_datasets(dataset_dir: Optional[str] = None) -> Union[dict, List[str]]: # TODO: remove union
    dataset_info = load_dataset_info(dataset_dir if dataset_dir is not None else DEFAULT_DATA_DIR)
    return gr.update(value=[], choices=list(dataset_info.keys())) if dataset_dir is not None else list(dataset_info.keys())
