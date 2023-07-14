import gradio as gr
from typing import Tuple
from gradio.components import Component

from glmtuner.webui.common import add_model, del_model, list_models, list_checkpoints


def create_model_manager(base_model: Component, model_list: Component) -> Component:
    with gr.Box(visible=False, elem_classes="modal-box") as model_manager:
        model_name = gr.Textbox(lines=1, label="Model name")
        model_path = gr.Textbox(lines=1, label="Model path", info="The absolute path to your model.")

        with gr.Row():
            confirm = gr.Button("Save")
            cancel = gr.Button("Cancel")

    confirm.click(
        add_model, [model_list, model_name, model_path], [model_list, model_name, model_path]
    ).then(
        lambda: gr.update(visible=False), outputs=[model_manager]
    ).then(
        list_models, [model_list], [base_model]
    )

    cancel.click(lambda: gr.update(visible=False), outputs=[model_manager])

    return model_manager


def create_model_tab() -> Tuple[Component, Component, Component]:

    model_list = gr.State([]) # gr.State does not accept a dict

    with gr.Row():
        base_model = gr.Dropdown(label="Model", interactive=True, scale=4)
        add_btn = gr.Button("Add model", scale=1)
        del_btn = gr.Button("Delete model", scale=1)

    with gr.Row():
        checkpoints = gr.Dropdown(label="Checkpoints", multiselect=True, interactive=True, scale=5)
        refresh = gr.Button("Refresh checkpoints", scale=1)

    model_manager = create_model_manager(base_model, model_list)

    base_model.change(list_checkpoints, [base_model], [checkpoints])

    add_btn.click(lambda: gr.update(visible=True), outputs=[model_manager]).then(
        list_models, [model_list], [base_model]
    )

    del_btn.click(del_model, [model_list, base_model], [model_list]).then(
        list_models, [model_list], [base_model]
    )

    refresh.click(list_checkpoints, [base_model], [checkpoints])

    return base_model, model_list, checkpoints
