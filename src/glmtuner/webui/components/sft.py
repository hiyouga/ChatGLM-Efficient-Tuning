import gradio as gr
from gradio.components import Component
from transformers.trainer_utils import SchedulerType

from glmtuner.webui.components.data import create_preview_box
from glmtuner.webui.common import list_datasets
from glmtuner.webui.runner import Runner
from glmtuner.webui.utils import can_preview, get_preview, get_time, gen_plot


def create_sft_tab(base_model: Component, model_list: Component, checkpoints: Component, runner: Runner) -> None:
    with gr.Row():
        finetuning_type = gr.Dropdown(
            label="Finetuning method", value="lora", choices=["full", "freeze", "p_tuning", "lora"], interactive=True
        )

    with gr.Row():
        dataset = gr.Dropdown(label="Dataset", choices=list_datasets(), multiselect=True, interactive=True, scale=4)
        preview = gr.Button("Preview", visible=False, scale=1)

    preview_box, preview_count, preview_samples = create_preview_box()

    dataset.change(can_preview, [dataset], [preview])
    preview.click(
        get_preview, [dataset], [preview_count, preview_samples]
    ).then(
        lambda: gr.update(visible=True), outputs=[preview_box]
    )

    with gr.Row():
        learning_rate = gr.Textbox(
            label="Learning rate", value="5e-5", info="The initial learning rate for AdamW.", interactive=True
        )
        num_train_epochs = gr.Textbox(
            label="Epochs", value="3.0", info="Total number of training epochs to perform.", interactive=True
        )
        max_samples = gr.Textbox(
            label="Max samples", value="100000", info="Number of samples for training.", interactive=True
        )
        use_v2 = gr.Checkbox(label="use ChatGLM2", value=True)

    with gr.Row():
        per_device_train_batch_size = gr.Slider(
            label="Batch size", value=4, minimum=1, maximum=128, step=1, info="Train batch size.", interactive=True
        )
        gradient_accumulation_steps = gr.Slider(
            label="Gradient accumulation", value=4, minimum=1, maximum=16, step=1, info='Accumulation steps.', interactive=True
        )
        lr_scheduler_type = gr.Dropdown(
            label="LR Scheduler", value="cosine", info="Scheduler type.",
            choices=[scheduler.value for scheduler in SchedulerType], interactive=True
        )
        fp16 = gr.Checkbox(label="fp16", value=True)

    with gr.Row():
        logging_steps = gr.Slider(
            label="Logging steps", value=1, minimum=1, maximum=1000, step=10,
            info="Number of update steps between two logs.", interactive=True
        )
        save_steps = gr.Slider(
            label="Save steps", value=100, minimum=10, maximum=2000, step=10,
            info="Number of updates steps before two checkpoint saves.", interactive=True
        )

    with gr.Row():
        start = gr.Button("Start training")
        stop = gr.Button("Abort")

    with gr.Row():
        with gr.Column(scale=4):
            output_dir = gr.Textbox(label="Checkpoint name", value=get_time(), interactive=True)
            output_info = gr.Markdown(value="Ready")

        with gr.Column(scale=1):
            loss_viewer = gr.Plot(label="Loss")

    start.click(
        runner.run_train,
        [
            base_model, model_list, checkpoints, output_dir, finetuning_type,
            dataset, learning_rate, num_train_epochs, max_samples,
            fp16, use_v2, per_device_train_batch_size, gradient_accumulation_steps,
            lr_scheduler_type, logging_steps, save_steps
        ],
        output_info
    )
    stop.click(runner.set_abort, queue=False)

    output_info.change(gen_plot, [base_model, output_dir], loss_viewer, queue=False)
