from typing import Dict
from transformers.trainer_utils import SchedulerType

import gradio as gr
from gradio.components import Component

from glmtuner.webui.common import list_datasets, DEFAULT_DATA_DIR, METHODS
from glmtuner.webui.components.data import create_preview_box
from glmtuner.webui.runner import Runner
from glmtuner.webui.utils import can_preview, get_preview, get_time, gen_plot


def create_sft_tab(top_elems: Dict[str, Component], runner: Runner) -> Dict[str, Component]:
    with gr.Row():
        finetuning_type = gr.Dropdown(value="lora", choices=METHODS, interactive=True, scale=2)
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, interactive=True, scale=1)
        dataset = gr.Dropdown(choices=list_datasets(), multiselect=True, interactive=True, scale=2)
        preview_btn = gr.Button(interactive=False, scale=1)

    preview_box, preview_count, preview_samples = create_preview_box()

    dataset_dir.change(list_datasets, [dataset_dir], [dataset])
    dataset.change(can_preview, [dataset_dir, dataset], [preview_btn])
    preview_btn.click(get_preview, [dataset_dir, dataset], [preview_count, preview_samples, preview_box])

    with gr.Row():
        learning_rate = gr.Textbox(value="5e-5", interactive=True)
        num_train_epochs = gr.Textbox(value="3.0", interactive=True)
        max_samples = gr.Textbox(value="100000", interactive=True)
        quantization_bit = gr.Dropdown([8, 4])

    with gr.Row():
        batch_size = gr.Slider(value=4, minimum=1, maximum=128, step=1, interactive=True)
        gradient_accumulation_steps = gr.Slider(value=4, minimum=1, maximum=32, step=1, interactive=True)
        lr_scheduler_type = gr.Dropdown(
            value="cosine", choices=[scheduler.value for scheduler in SchedulerType], interactive=True
        )
        fp16 = gr.Checkbox(value=True)

    with gr.Row():
        logging_steps = gr.Slider(value=5, minimum=5, maximum=1000, step=5, interactive=True)
        save_steps = gr.Slider(value=100, minimum=10, maximum=2000, step=10, interactive=True)

    with gr.Row():
        start_btn = gr.Button()
        stop_btn = gr.Button()

    with gr.Row():
        with gr.Column(scale=4):
            output_dir = gr.Textbox(value=get_time(), interactive=True)
            output = gr.Markdown()

        with gr.Column(scale=1):
            loss_viewer = gr.Plot()

    start_btn.click(
        runner.run_train,
        [
            top_elems["model_name"], top_elems["model_path"], top_elems["checkpoints"],
            output_dir, finetuning_type,
            dataset, dataset_dir, learning_rate, num_train_epochs, max_samples,
            fp16, quantization_bit, batch_size, gradient_accumulation_steps,
            lr_scheduler_type, logging_steps, save_steps
        ],
        output
    )
    stop_btn.click(runner.set_abort, queue=False)

    output.change(gen_plot, [top_elems["model_name"], output_dir], loss_viewer, queue=False)

    return dict(
        finetuning_type=finetuning_type,
        dataset_dir=dataset_dir,
        dataset=dataset,
        preview_btn=preview_btn,
        preview_count=preview_count,
        preview_samples=preview_samples,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_samples=max_samples,
        quantization_bit=quantization_bit,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        fp16=fp16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        start_btn=start_btn,
        stop_btn=stop_btn,
        output_dir=output_dir,
        output=output,
        loss_viewer=loss_viewer
    )
