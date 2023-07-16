from typing import Dict
import gradio as gr
from gradio.components import Component

from glmtuner.webui.common import list_datasets, DEFAULT_DATA_DIR
from glmtuner.webui.components.data import create_preview_box
from glmtuner.webui.runner import Runner
from glmtuner.webui.utils import can_preview, get_preview


def create_eval_tab(top_elems: Dict[str, Component], runner: Runner) -> Dict[str, Component]:
    with gr.Row():
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, interactive=True, scale=2)
        dataset = gr.Dropdown(choices=list_datasets(), multiselect=True, interactive=True, scale=4)
        preview_btn = gr.Button(interactive=False, scale=1)

    preview_box, preview_count, preview_samples = create_preview_box()

    dataset_dir.change(list_datasets, [dataset_dir], [dataset])
    dataset.change(can_preview, [dataset_dir, dataset], [preview_btn])
    preview_btn.click(get_preview, [dataset_dir, dataset], [preview_count, preview_samples, preview_box])

    with gr.Row():
        max_samples = gr.Textbox(value="100000", interactive=True)
        eval_batch_size = gr.Slider(value=8, minimum=1, maximum=128, step=1, interactive=True)
        quantization_bit = gr.Dropdown([8, 4])

    with gr.Row():
        start_btn = gr.Button()
        stop_btn = gr.Button()

    output = gr.Markdown()

    start_btn.click(
        runner.run_eval,
        [
            top_elems["model_name"], top_elems["model_path"], top_elems["checkpoints"],
            dataset, dataset_dir, max_samples, eval_batch_size, quantization_bit
        ],
        [output]
    )
    stop_btn.click(runner.set_abort, queue=False)

    return dict(
        dataset_dir=dataset_dir,
        # dataset=dataset,
        # preview_btn=preview_btn,
        # preview_count=preview_count,
        # preview_samples=preview_samples,
        # max_samples=max_samples,
        # eval_batch_size=eval_batch_size,
        # quantization_bit=quantization_bit,
        # start_btn=start_btn,
        # stop_btn=stop_btn,
        # output=output
    )
