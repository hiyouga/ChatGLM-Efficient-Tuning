import gradio as gr
import json
from webui import (
    common,
    interface
)

def main():
    with open(f"{common.css_dir}/main.css") as f:
        css = f.read()
    with gr.Blocks(css=css) as demo:
        interface.create_sft_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)


if __name__ == "__main__":
    main()
