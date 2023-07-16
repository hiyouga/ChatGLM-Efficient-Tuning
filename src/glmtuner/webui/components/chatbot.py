from typing import Tuple

import gradio as gr
from gradio.blocks import Block
from gradio.components import Component

from glmtuner.webui.chat import WebChatModel


def create_chat_box(chat_model: WebChatModel) -> Tuple[Block, Component, Component]:
    with gr.Box(visible=False) as chat_box:
        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    query = gr.Textbox(show_label=False, placeholder="Input...", lines=10)

                with gr.Column(min_width=32, scale=1):
                    submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                clear_btn = gr.Button("Clear History")
                max_length = gr.Slider(
                    10, 2048, value=chat_model.generating_args.max_length, step=1.0,
                    label="Maximum length", interactive=True
                )
                top_p = gr.Slider(
                    0, 1, value=chat_model.generating_args.top_p, step=0.01,
                    label="Top P", interactive=True
                )
                temperature = gr.Slider(
                    0, 1.5, value=chat_model.generating_args.temperature, step=0.01,
                    label="Temperature", interactive=True
                )

    history = gr.State([])

    submit_btn.click(
        chat_model.predict,
        [chatbot, query, history, max_length, top_p, temperature],
        [chatbot, history],
        show_progress=True
    ).then(
        lambda: gr.update(value=""), outputs=[query]
    )

    clear_btn.click(lambda: ([], []), outputs=[chatbot, history], show_progress=True)

    return chat_box, chatbot, history
