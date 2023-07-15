from typing import Tuple

import gradio as gr
from gradio.components import Component

from glmtuner.extras.constants import SUPPORTED_MODEL_LIST
from glmtuner.webui.common import list_checkpoints, load_temp_use_config, save_model_config


def create_model_tab() -> Tuple[Component, Component, Component]:
    user_config = load_temp_use_config()
    gr_state = gr.State([])  # gr.State does not accept a dict

    with gr.Row():
        model_name = gr.Dropdown([model["pretrained_model_name"] for model in SUPPORTED_MODEL_LIST] + ["custom"],
                                 label="Base Model", info="Model Version of ChatGLM",
                                 value=user_config.get("model_name"))
        model_path = gr.Textbox(lines=1, label="Local model path(Optional)",
                                info="The absolute path of the directory where the local model file is located",
                                value=user_config.get("model_path"))

    with gr.Row():
        checkpoints = gr.Dropdown(label="Checkpoints", multiselect=True, interactive=True, scale=5)
        refresh = gr.Button("Refresh checkpoints", scale=1)

    model_name.change(list_checkpoints, [model_name], [checkpoints])
    model_path.change(save_model_config, [model_name, model_path])
    refresh.click(list_checkpoints, [model_name], [checkpoints])

    return model_name, model_path, checkpoints
