from typing import Dict

import gradio as gr
from gradio.components import Component

from glmtuner.extras.constants import SUPPORTED_MODELS
from glmtuner.webui.common import list_checkpoints, load_config, save_config


def create_top() -> Dict[str, Component]:
    user_config = load_config() # TODO: use model list
    available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]

    with gr.Row():
        lang = gr.Dropdown(choices=["en", "zh"], value="en", interactive=True, scale=1)
        model_name = gr.Dropdown(choices=available_models, value=user_config.get("model_name", None), scale=2)
        model_path = gr.Textbox(value=user_config.get("model_path", None), scale=2)

    with gr.Row():
        checkpoints = gr.Dropdown(multiselect=True, interactive=True, scale=5)
        refresh_btn = gr.Button(scale=1)

    model_name.change(
        list_checkpoints, [model_name], [checkpoints]
    ).then( # TODO: save model list
        lambda: gr.update(value=""), outputs=[model_path]
    )
    model_path.change(save_config, [model_name, model_path])
    refresh_btn.click(list_checkpoints, [model_name], [checkpoints])

    return dict(
        lang=lang,
        model_name=model_name,
        model_path=model_path,
        checkpoints=checkpoints,
        refresh_btn=refresh_btn
    )
