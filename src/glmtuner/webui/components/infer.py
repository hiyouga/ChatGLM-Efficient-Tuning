from typing import Tuple

import gradio as gr
from gradio.components import Component

from glmtuner.webui.chat import WebChatModel


def create_chat_box(chat_model: WebChatModel) -> Tuple[Component, Component, Component]:
    with gr.Box(visible=False) as chat_box:
        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    query = gr.Textbox(show_label=False, placeholder="Input...", lines=10)

                with gr.Column(min_width=32, scale=1):
                    submit = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                clear = gr.Button("Clear History")
                max_length = gr.Slider(
                    10, 2048, value=chat_model.generating_args.max_length, step=1.0, label="Maximum length",
                    interactive=True
                )
                top_p = gr.Slider(
                    0, 1, value=chat_model.generating_args.top_p, step=0.01, label="Top P", interactive=True
                )
                temperature = gr.Slider(
                    0, 1.5, value=chat_model.generating_args.temperature, step=0.01, label="Temperature",
                    interactive=True
                )

    history = gr.State([])

    submit.click(
        chat_model.predict,
        [chatbot, query, history, max_length, top_p, temperature],
        [chatbot, history],
        show_progress=True
    ).then(
        lambda: gr.update(value=""), outputs=[query]
    )

    clear.click(lambda: ([], []), outputs=[chatbot, history], show_progress=True)

    return chat_box, chatbot, history


def create_infer_tab(base_model: Component, model_path: Component, checkpoints: Component) -> None:
    info_box = gr.Markdown(value="Model unloaded, please load a model first.")

    chat_model = WebChatModel()
    chat_box, chatbot, history = create_chat_box(chat_model)

    with gr.Row():
        load_btn = gr.Button("Load model")
        unload_btn = gr.Button("Unload model")
        quantization_bit = gr.Dropdown([8, 4], label="Quantization bit", info="Only support 4 bit or 8 bit")

    load_btn.click(
        chat_model.load_model, [base_model, model_path, checkpoints, quantization_bit], [info_box]
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
