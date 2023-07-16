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
        learning_rate = gr.Textbox(
            label="Learning rate", value="5e-5", info="The initial learning rate for AdamW.", interactive=True
        )
        num_train_epochs = gr.Textbox(
            label="Epochs", value="3.0", info="Total number of training epochs to perform.", interactive=True
        )
        max_samples = gr.Textbox(
            label="Max samples", value="100000", info="Samples to use.", interactive=True
        )
        quantization_bit = gr.Dropdown([8, 4], label="Quantization bit", info="Quantize model to 4/8-bit mode.")

    with gr.Row():
        train_batch_size = gr.Slider(
            label="Batch size", value=4, minimum=1, maximum=128, step=1,
            info="Train batch size.", interactive=True
        )
        gradient_accumulation_steps = gr.Slider(
            label="Gradient accumulation", value=4, minimum=1, maximum=32, step=1,
            info="Accumulation steps.", interactive=True
        )
        lr_scheduler_type = gr.Dropdown(
            label="LR Scheduler", value="cosine", info="Scheduler type.",
            choices=[scheduler.value for scheduler in SchedulerType], interactive=True
        )
        fp16 = gr.Checkbox(label="fp16", value=True)

    with gr.Row():
        logging_steps = gr.Slider(
            label="Logging steps", value=5, minimum=5, maximum=1000, step=5,
            info="Number of update steps between two logs.", interactive=True
        )
        save_steps = gr.Slider(
            label="Save steps", value=100, minimum=10, maximum=2000, step=10,
            info="Number of updates steps before two checkpoint saves.", interactive=True
        )

    with gr.Row():
        start_btn = gr.Button("Start training")
        stop_btn = gr.Button("Abort")

    with gr.Row():
        with gr.Column(scale=4):
            output_dir = gr.Textbox(label="Checkpoint name", value=get_time(), interactive=True)
            output_info = gr.Markdown(value="Ready")

        with gr.Column(scale=1):
            loss_viewer = gr.Plot(label="Loss")

    start_btn.click(
        runner.run_train,
        [
            top_elems["model_name"], top_elems["model_path"], top_elems["checkpoints"],
            output_dir, finetuning_type,
            dataset, dataset_dir, learning_rate, num_train_epochs, max_samples,
            fp16, quantization_bit, train_batch_size, gradient_accumulation_steps,
            lr_scheduler_type, logging_steps, save_steps
        ],
        output_info
    )
    stop_btn.click(runner.set_abort, queue=False)

    output_info.change(gen_plot, [top_elems["model_name"], output_dir], loss_viewer, queue=False)

    return dict()
