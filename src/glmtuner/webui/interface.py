import gradio as gr
from transformers.utils.versions import require_version

from glmtuner.webui.css import CSS
from glmtuner.webui.runner import Runner
from glmtuner.webui.components import (
    create_model_tab,
    create_sft_tab,
    create_eval_tab,
    create_infer_tab
)


require_version("gradio>=3.36.0", "To fix: pip install gradio>=3.36.0")


def create_ui() -> gr.Blocks:
    runner = Runner()

    with gr.Blocks(title="Web Tuner", css=CSS) as demo:
        base_model, model_list, checkpoints = create_model_tab()

        with gr.Tab("SFT"):
            create_sft_tab(base_model, model_list, checkpoints, runner)

        with gr.Tab("Evaluate"):
            create_eval_tab(base_model, model_list, checkpoints, runner)

        with gr.Tab("Inference"):
            create_infer_tab(base_model, model_list, checkpoints)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)
