from typing import Dict

import gradio as gr
from gradio.components import Component

from glmtuner.webui.chat import WebChatModel
from glmtuner.webui.components.chatbot import create_chat_box


def create_infer_tab(top_elems: Dict[str, Component]) -> Dict[str, Component]:
    info_box = gr.Markdown(value="Model unloaded, please load a model first.")

    chat_model = WebChatModel()
    chat_box, chatbot, history = create_chat_box(chat_model)

    with gr.Row():
        load_btn = gr.Button("Load model")
        unload_btn = gr.Button("Unload model")
        quantization_bit = gr.Dropdown([8, 4], label="Quantization bit", info="Quantize model to 4/8-bit mode.")

    load_btn.click(
        chat_model.load_model,
        [top_elems["model_name"], top_elems["model_path"], top_elems["checkpoints"], quantization_bit],
        [info_box]
    ).then(
        lambda: gr.update(visible=(chat_model.model is not None)), outputs=[chat_box]
    )

    unload_btn.click(
        chat_model.unload_model, outputs=[info_box]
    ).then(
        lambda: ([], []), outputs=[chatbot, history]
    ).then(
        lambda: gr.update(visible=(chat_model.model is not None)), outputs=[chat_box]
    )

    return dict()
