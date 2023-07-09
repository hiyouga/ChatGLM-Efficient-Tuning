import gradio as gr
from webui import (
    ui,
    plot,
    common,
    runner
)
import json

def get_available_model():
    return list(common.settings["path_to_model"].keys())

def get_available_dataset():
    with open(f'{common.data_dir}/dataset_info.json') as f:
        dataset_info = json.load(f)
        return list(dataset_info.keys())

def get_available_ckpt():
    print('refresh ckpt...')
    return []

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

def create_sft_interface(base_model):
    process = runner.Runner()
    
    with gr.Tab('Train', elem_id='train-tab'):
        with gr.Row():
            ft_type = gr.Dropdown(label='Which fine-tuning method to use.', value='lora', choices=['none', 'freeze', 'lora', 'full'], interactive=True)
        with gr.Row():
            dataset = gr.Dropdown(label='Dataset', info='The name of provided dataset(s) to use. Use comma to separate multiple datasets.', choices=get_available_dataset(), interactive=True)
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
                ckpt = gr.Textbox(label="Checkpoint Name", value="logs/test", interactive=False)
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
            base_model.change(get_available_ckpt, None, None)
        with gr.Row():
            eval_dataset = gr.Dropdown(label='Dataset', info='The name of provided dataset(s) to use. Use comma to separate multiple datasets.', choices=get_available_dataset(), interactive=True)
        with gr.Row():
            per_device_eval_batch_size = gr.Slider(label='Batch Size', value=8, minimum=1, maximum=128, step=1, info='Batch size per GPU/TPU core/CPU for evaluation.', interactive=True)
        eval_start_btn = gr.Button("Start Evaluation")
        eval_output = gr.Markdown(value="Ready")
        eval_start_btn.click(process.run_eval, [base_model, checkpoint, eval_dataset, per_device_eval_batch_size], eval_output)

def create_infer_interface(base_model):
    with gr.Row():
        checkpoint = gr.Dropdown(label='Checkpoint', choices=get_available_ckpt(), interactive=True)
        ui.create_refresh_button(checkpoint, lambda: None, lambda: {'choices': get_available_ckpt()}, 'emoji-button')

def create_interface():
    with open(f"{common.css_dir}/main.css") as f:
        css = f.read()
    with gr.Blocks(css=css) as demo:
        base_model = create_model_tab()
        with gr.Tab('Train', elem_id='sft-tab'):
            create_sft_interface(base_model)
        with gr.Tab('Infer', elem_id='infer_tab'):
            create_infer_interface(base_model)
    return demo
