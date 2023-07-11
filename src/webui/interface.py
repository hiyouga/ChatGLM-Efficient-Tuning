import gradio as gr
from webui import (
    ui,
    plot,
    common,
    runner,
    chat,
    utils
)
import json
import os

def get_available_model():
    return list(common.settings["path_to_model"].keys())

def get_available_dataset():
    with open(f'{common.data_dir}/dataset_info.json') as f:
        dataset_info = json.load(f)
        return list(dataset_info.keys())

def update_preview_btn(dname):
    with open(f'{common.data_dir}/dataset_info.json') as f:
        dataset_info = json.load(f)
    return gr.update(visible="file_name" in dataset_info[dname])

def get_dataset_preview(dname):
    with open(f'{common.data_dir}/dataset_info.json') as f:
        dataset_info = json.load(f)
    dpath = dataset_info[dname]["file_name"]
    with open(f'{common.data_dir}/{dpath}') as f:
        data = json.load(f)
    return data[:2], len(data)

def get_available_ckpt():
    ckpts = []
    save_path = common.get_save_dir()
    if save_path and os.path.isdir(save_path):
        for ckpt_dir in os.listdir(save_path):
            if os.path.isdir(os.path.join(save_path, ckpt_dir)) and os.path.isfile(os.path.join(save_path, ckpt_dir, 'adapter_model.bin')):
                ckpts.append(ckpt_dir)
    return ckpts

def create_model_tab():
    with gr.Row():
        base_model = gr.Dropdown(label='Base model', value=common.settings['base_model'], choices=get_available_model(), interactive=True)
        ui.create_refresh_button(base_model, lambda: None, lambda: {'choices': get_available_model()}, 'emoji-button')
        save_btn = gr.Button('💾', elem_classes=['emoji-button'])
        del_btn = gr.Button('🗑️', elem_classes=['emoji-button'])
        base_model.change(common.set_base_model, [base_model], None)

    with gr.Box(visible=False, elem_classes='model-saver') as save_box:
        model_name = gr.Textbox(lines=1, label='Model name')
        model_path = gr.Textbox(lines=1, label='Model path', info='The absolute path to your model.')
    
        with gr.Row():
            confirm_btn = gr.Button('Save', elem_classes="small-button")
            cancel_btn = gr.Button('Cancel', elem_classes="small-button")
    
    confirm_btn.click(common.add_base_model, [model_name, model_path], None).then(
        lambda: gr.update(visible=False), None, save_box
    )
    cancel_btn.click(lambda: gr.update(visible=False), None, save_box)

    save_btn.click(lambda: gr.update(visible=True), None, save_box).then(
        lambda: gr.update(choices=get_available_model()), None, base_model
    )
    del_btn.click(common.del_base_model, [base_model], None).then(
        lambda: gr.update(choices=get_available_model()), None, base_model
    )

    return base_model

def create_preview_box():
    with gr.Box(visible=False, elem_classes='model-saver') as preview:
        with gr.Row():
            preview_count = gr.Number(label='Number of Data', interactive=False)
        with gr.Row():
            preview_content = gr.JSON(label='Sample', interactive=False)
        close_btn = gr.Button('Close', elem_classes="small-button")
    close_btn.click(lambda: gr.update(visible=False), None, preview)
    return preview, preview_count, preview_content

def create_sft_interface(base_model):
    process = runner.Runner()
    
    with gr.Tab('Train', elem_id='train-tab'):
        with gr.Row():
            ft_type = gr.Dropdown(label='Which fine-tuning method to use.', value='lora', choices=['none', 'freeze', 'lora', 'full'], interactive=True)
        with gr.Row():
            dataset = gr.Dropdown(label='Dataset', info='The name of provided dataset(s) to use. Use comma to separate multiple datasets.', choices=get_available_dataset(), interactive=True)
            preview_btn = gr.Button('🔍', elem_classes=['emoji-button'], visible=False)
        dataset.change(update_preview_btn, [dataset], [preview_btn])
        preview, preview_count, preview_content = create_preview_box()
        preview_btn.click(get_dataset_preview, [dataset], [preview_content, preview_count]).then(
            lambda: gr.update(visible=True), None, preview
        )
        with gr.Row():
            learning_rate = gr.Textbox(label='Learning Rate', value='5e-5', info='The initial learning rate for AdamW.', interactive=True)
            num_train_epochs = gr.Textbox(label='Epochs', value='3.0', info='Total number of training epochs to perform.', interactive=True)
            fp16 = gr.Checkbox(label="fp16", value=True)
        with gr.Row():
            per_device_train_batch_size = gr.Slider(label='Batch Size', value=4, minimum=1, maximum=128, step=1, info='Batch size per GPU/TPU core/CPU for training.', interactive=True)
            gradient_accumulation_steps = gr.Slider(label='Gradient Accumulation', value=4, minimum=1, maximum=16, step=1, info='Number of updates steps to accumulate before performing a backward/update pass.', interactive=True)
            lr_scheduler_type = gr.Dropdown(label='LR Scheduler', value='cosine', info='The scheduler type to use.', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau'], interactive=True)
        with gr.Row():
            logging_steps = gr.Slider(label='Logging Steps', value=10, minimum=5, maximum=500, step=5, info='Log every X updates steps. Should be an integer or a float in range `[0,1)`.If smaller than 1, will be interpreted as ratio of total training steps.', interactive=True)
            save_steps = gr.Slider(label='Save Steps', value=1000, minimum=100, maximum=2000, step=100, info='Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`.If smaller than 1, will be interpreted as ratio of total training steps.', interactive=True)

        with gr.Row():
            start_btn = gr.Button("Start training")
            stop_btn = gr.Button("Abort")

        # output demo
        with gr.Row():
            with gr.Column(scale=4):
                ckpt = gr.Textbox(label="Checkpoint Name", value=utils.generate_ckpt_name, interactive=False)
                output = gr.Markdown(value="Ready")
            with gr.Column(scale=1):
                with gr.Row():
                    loss_plt = gr.Plot(label="Loss")
        start_btn.click(process.run_train, [ckpt, base_model, ft_type, dataset, learning_rate, num_train_epochs, fp16, per_device_train_batch_size, gradient_accumulation_steps, lr_scheduler_type, logging_steps, save_steps], output)
        stop_btn.click(process.set_abort, None, None, queue=False)

        output.change(plot.plt_loss, [ckpt], loss_plt, queue=False)

    # eval
    with gr.Tab('Evaluation', elem_id='eval-tab'):
        with gr.Row():
            checkpoint = gr.Dropdown(label='Checkpoint', choices=get_available_ckpt(), interactive=True)
            ui.create_refresh_button(checkpoint, lambda: None, lambda: {'choices': get_available_ckpt()}, 'emoji-button')
            base_model.change(lambda: gr.update(choices=get_available_ckpt()), None, checkpoint)
        with gr.Row():
            eval_dataset = gr.Dropdown(label='Dataset', info='The name of provided dataset(s) to use. Use comma to separate multiple datasets.', choices=get_available_dataset(), interactive=True)
        with gr.Row():
            per_device_eval_batch_size = gr.Slider(label='Batch Size', value=8, minimum=1, maximum=128, step=1, info='Batch size per GPU/TPU core/CPU for evaluation.', interactive=True)
        with gr.Row():
            eval_start_btn = gr.Button("Start Evaluation")
            eval_stop_btn = gr.Button("Abort")
        eval_output = gr.Markdown(value="Ready")

        eval_start_btn.click(process.run_eval, [base_model, checkpoint, eval_dataset, per_device_eval_batch_size], eval_output)
        eval_stop_btn.click(process.set_abort, None, None, queue=False)

def create_chat_box(chater):
    with gr.Box(visible=False) as chat_box:
        chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 4096, value=chater.generating_args.max_length, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=chater.generating_args.top_p, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1.5, value=chater.generating_args.temperature, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(chater.predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history], show_progress=True)
    submitBtn.click(lambda: gr.update(value=''), [], [user_input])

    emptyBtn.click(lambda: ([], []), outputs=[chatbot, history], show_progress=True)

    return chat_box, chatbot, history

def create_infer_interface(base_model):
    chater = chat.Chater()
    with gr.Row():
        checkpoint = gr.Dropdown(label='Checkpoint', choices=get_available_ckpt(), interactive=True)
        base_model.change(lambda: gr.update(choices=get_available_ckpt()), None, checkpoint)
        ui.create_refresh_button(checkpoint, lambda: None, lambda: {'choices': get_available_ckpt()}, 'emoji-button')
    with gr.Row():
        load_btn = gr.Button('load', elem_classes="small-button")
        unload_btn = gr.Button('unload', elem_classes="small-button")
    infer_info = gr.Markdown(value='Model unloaded, please load a model first.')

    chat_box, chatbot, history = create_chat_box(chater)

    load_btn.click(chater.load_model, [base_model, checkpoint], [infer_info]).then(
        lambda: gr.update(visible=True), None, chat_box
    )
    unload_btn.click(chater.unload_model, None, [infer_info]).then(
        lambda: ([], []), outputs=[chatbot, history]).then(
        lambda: gr.update(visible=False), None, chat_box
    )


def create_interface():
    with open(f"{common.css_dir}/main.css") as f:
        css = f.read()
    with gr.Blocks(css=css) as demo:
        base_model = create_model_tab()
        with gr.Tab('Train-SFT', elem_id='sft-tab'):
            create_sft_interface(base_model)
        with gr.Tab('Infer', elem_id='infer_tab'):
            create_infer_interface(base_model)
    return demo
