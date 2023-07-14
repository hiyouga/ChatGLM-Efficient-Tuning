import gradio as gr
from gradio.components import Component

from glmtuner.webui.common import list_datasets
from glmtuner.webui.runner import Runner


def create_eval_tab(base_model: Component, model_list: Component, checkpoints: Component, runner: Runner) -> None:
    with gr.Row():
        dataset = gr.Dropdown(
            label="Dataset", info="The name of dataset(s).", choices=list_datasets(), multiselect=True, interactive=True
        )

    with gr.Row():
        max_samples = gr.Textbox(
            label="Max samples", value="100000", info="Number of samples for training.", interactive=True
        )
        per_device_eval_batch_size = gr.Slider(
            label="Batch size", value=8, minimum=1, maximum=128, step=1, info="Eval batch size.", interactive=True
        )
        use_v2 = gr.Checkbox(label="use ChatGLM2", value=True)

    with gr.Row():
        start = gr.Button("Start evaluation")
        stop = gr.Button("Abort")

    output = gr.Markdown(value="Ready")

    start.click(
        runner.run_eval,
        [base_model, model_list, checkpoints, dataset, max_samples, per_device_eval_batch_size, use_v2],
        [output]
    )
    stop.click(runner.set_abort, queue=False)
