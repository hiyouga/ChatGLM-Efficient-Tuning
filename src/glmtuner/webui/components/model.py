from typing import Tuple

import gradio as gr
from gradio.components import Component

from glmtuner.extras.constants import SUPPORTED_MODELS
from glmtuner.webui.common import list_checkpoints, load_config, save_config


def create_model_tab() -> Tuple[Component, Component, Component]:
    user_config = load_config()
    available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]

    with gr.Row():
        model_name = gr.Dropdown(choices=available_models, label="Model", value=user_config.get("model_name", None))
        model_path = gr.Textbox(
            label="Local path (Optional)", value=user_config.get("model_path", None),
            info="The absolute path of the directory where the local model file is located."
        )

    with gr.Row():
        checkpoints = gr.Dropdown(label="Checkpoints", multiselect=True, interactive=True, scale=5)
        refresh_btn = gr.Button("Refresh checkpoints", scale=1)

    model_name.change(
        list_checkpoints, [model_name], [checkpoints]
    ).then( # TODO: save list
        lambda: gr.update(value=""), outputs=[model_path]
    )
    model_path.change(save_config, [model_name, model_path])
    refresh_btn.click(list_checkpoints, [model_name], [checkpoints])

    return model_name, model_path, checkpoints
