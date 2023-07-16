import gradio as gr
from gradio.components import Component

from glmtuner.webui.common import list_datasets
from glmtuner.webui.components.data import create_preview_box
from glmtuner.webui.runner import Runner
from glmtuner.webui.utils import can_preview, get_preview


def create_eval_tab(model_name: Component, model_path: Component, checkpoints: Component, runner: Runner) -> None:
    with gr.Row():
        dataset = gr.Dropdown(label="Dataset", choices=list_datasets(), multiselect=True, interactive=True, scale=4)
        preview_btn = gr.Button("Preview", interactive=False, scale=1)

    preview_box, preview_count, preview_samples = create_preview_box()

    dataset.change(can_preview, [dataset], [preview_btn])
    preview_btn.click(
        get_preview, [dataset], [preview_count, preview_samples]
    ).then(
        lambda: gr.update(visible=True), outputs=[preview_box]
    )

    with gr.Row():
        max_samples = gr.Textbox(
            label="Max samples", value="100000", info="Number of samples for training.", interactive=True
        )
        per_device_eval_batch_size = gr.Slider(
            label="Batch size", value=8, minimum=1, maximum=128, step=1, info="Eval batch size.", interactive=True
        )
        quantization_bit = gr.Dropdown([8, 4], label="Quantization bit", info="Quantize model to 4/8-bit mode.")

    with gr.Row():
        start_btn = gr.Button("Start evaluation")
        stop_btn = gr.Button("Abort")

    output = gr.Markdown(value="Ready")

    start_btn.click(
        runner.run_eval,
        [model_name, model_path, checkpoints, dataset, max_samples, per_device_eval_batch_size, quantization_bit],
        [output]
    )
    stop_btn.click(runner.set_abort, queue=False)
