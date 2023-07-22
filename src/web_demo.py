# coding=utf-8
# Implements user interface in browser for ChatGLM fine-tuned with PEFT.
# Usage: python web_demo.py --checkpoint_dir path_to_checkpoint [--quantization_bit 4]

import gradio as gr
from transformers.utils.versions import require_version

from glmtuner.tuner import get_infer_args
from glmtuner.webui.chat import WebChatModel
from glmtuner.webui.components.chatbot import create_chat_box
from glmtuner.webui.manager import Manager


require_version("gradio>=3.36.0", "To fix: pip install gradio>=3.36.0")


def main():
    chat_model = WebChatModel(*get_infer_args())

    with gr.Blocks(title="Web Demo") as demo:
        lang = gr.Dropdown(choices=["en", "zh"], value="en")

        _, _, _, chat_elems = create_chat_box(chat_model, visible=True)

        manager = Manager([{"lang": lang}, chat_elems])

        demo.load(manager.gen_label, [lang], [lang] + list(chat_elems.values()))

        lang.change(manager.gen_label, [lang], [lang] + list(chat_elems.values()))

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
