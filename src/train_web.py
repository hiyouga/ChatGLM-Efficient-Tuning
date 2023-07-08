import gradio as gr
import torch
import json
import time
import threading
import transformers
import logging
from pet import (
    get_train_args,
    get_infer_args,
    run_sft
)
from webui.plot import plt_loss
from datetime import timedelta

path_to_model = {"llama-7b": "/home/incoming/zhengyw/llama/7b", "chatglm1": "/home/incoming/zhengyw/chatglm1"}

BASE_MODEL = None
ABORTED = False
def set_abort():
    global ABORTED
    ABORTED = True

def set_base_model(model_name):
    global BASE_MODEL
    BASE_MODEL = model_name

def run_train(model_name, base_model, ft_type, dataset, lr, epochs, fp16, per_device_train_batch_size, gradient_accumulation_steps, lr_scheduler_type, logging_steps, save_steps):
    global ABORTED
    ABORTED = False
    args = {
        "model_name_or_path": path_to_model[base_model],
        "do_train": True,
        "dataset": dataset,
        "finetuning_type": ft_type,
        "output_dir": model_name,
        "overwrite_cache": True,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lr_scheduler_type": lr_scheduler_type,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "learning_rate": float(lr),
        "num_train_epochs": float(epochs),
        "fp16": fp16,
    }
    model_args, data_args, training_args, finetuning_args, general_args = get_train_args(args)
    class log_handler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.log = ""

        def emit(self, record):
            if record.name == "httpx":
                return
            log_entry = self.format(record)
            self.callback(log_entry)

        def callback(self, log_entry):
            self.log += log_entry
            self.log += '\n\n'
    callback_handler = log_handler()
    callback_handler.setLevel(logging.INFO)
    logging.root.addHandler(callback_handler)


    class Callbacks(transformers.TrainerCallback):
        def __init__(self):
            self.start_time = time.time()
            tracker_key = ["current_steps", "total_steps", "loss", "reward", "learning_rate", "epoch", "percentage", "elapsed_time", "remaining_time"]
            self.tracker = {k: None for k in tracker_key}

        def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            if ABORTED:
                control.should_epoch_stop = True
                control.should_training_stop = True
        
        def on_substep_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            if ABORTED:
                control.should_epoch_stop = True
                control.should_training_stop = True
        
        def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, logs, **kwargs):
            if "loss" not in state.log_history[-1]:
                return
            cur_time = time.time()
            cur_steps = state.log_history[-1].get("step")
            elapsed_time = cur_time - self.start_time
            avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
            remaining_steps = state.max_steps - cur_steps
            remaining_time = remaining_steps * avg_time_per_step
            self.tracker = {
                "current_steps": cur_steps,
                "total_steps": state.max_steps,
                "loss": state.log_history[-1].get("loss", None),
                "reward": state.log_history[-1].get("reward", None),
                "learning_rate": state.log_history[-1].get("learning_rate", None),
                "epoch": state.log_history[-1].get("epoch", None),
                "percentage": round(cur_steps / state.max_steps * 100, 2) if state.max_steps != 0 else 100,
                "elapsed_time": str(timedelta(seconds=int(elapsed_time))),
                "remaining_time": str(timedelta(seconds=int(remaining_time)))
            }
    train_callback = Callbacks()
    def thread_run():
        run_sft(model_args, data_args, training_args, finetuning_args, [train_callback])
    
    thread = threading.Thread(target=thread_run)
    thread.start()

    def format_info(trainer_log, tracker):
        ret_info = trainer_log
        if tracker["current_steps"]:
            ret_info += '\n'
            ret_info += f"Running... **{tracker['current_steps']} / {tracker['total_steps']}** ... {tracker['elapsed_time']} < {tracker['remaining_time']}"
        return ret_info

    while thread.is_alive():
        time.sleep(1)
        if ABORTED:
            yield "Aborted, wait for terminating..."
        else:
            yield format_info(callback_handler.log, train_callback.tracker)
    
    yield "Ready"


def run_eval(base_model, ckpt, dataset, per_device_eval_batch_size):
    args = {
        "model_name_or_path": base_model,
        "do_eval": True,
        "dataset": dataset,
        "checkpoint_dir": ckpt,
        "output_dir": f"web_log/{ckpt}_eval",
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "predict_with_generate": True
    }
    model_args, data_args, training_args, finetuning_args, general_args = get_train_args(args)
    # run_sft(model_args, data_args, training_args, finetuning_args)
    raise NotImplementedError


class ToolButton(gr.Button, gr.components.IOComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_block_name(self):
        return "button"

def get_available_model():
    return path_to_model.keys()

def get_available_dataset():
    with open('data/dataset_info.json') as f:
        dataset_info = json.load(f)
        return dataset_info.keys()

def get_available_ckpt():
    print('refresh ckpt...')
    return []

refresh_symbol = '🔄'
def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_class):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_classes=elem_class)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )

    return refresh_button

def create_sft_interface():
    with gr.Row():
        base_model = gr.Dropdown(label='Base model', value='None', choices=get_available_model(), interactive=True)
        # create_refresh_button(base_model, lambda: None, lambda: {'choices': get_available_model()}, 'emoji-button')
        base_model.change(set_base_model, [base_model], None)
    with gr.Tab('Train', elem_id='train-tab'):
        with gr.Row():
            model_name = gr.Textbox(label='Name', info='The name of your new model', interactive=True)
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
                output = gr.Markdown(value="Ready")
            with gr.Column(scale=1):
                with gr.Row():
                    loss_plt = gr.Plot(label="Loss")
                with gr.Row():
                    loss_btn = gr.Button("Refresh Loss")
        start_btn.click(run_train, [model_name, base_model, ft_type, dataset, learning_rate, num_train_epochs, fp16, per_device_train_batch_size, gradient_accumulation_steps, lr_scheduler_type, logging_steps, save_steps], output)
        stop_btn.click(set_abort, None, None, queue=False)
        loss_btn.click(plt_loss, None, loss_plt)

    # eval
    with gr.Tab('Evaluation', elem_id='eval-tab'):
        with gr.Row():
            checkpoint = gr.Dropdown(label='Checkpoint', choices=get_available_ckpt(), interactive=True)
            create_refresh_button(checkpoint, lambda: None, lambda: {'choices': get_available_ckpt()}, 'emoji-button')
            base_model.change(get_available_ckpt, None, None)
        with gr.Row():
            eval_dataset = gr.Dropdown(label='Dataset', info='The name of provided dataset(s) to use. Use comma to separate multiple datasets.', choices=get_available_dataset(), interactive=True)
        with gr.Row():
            per_device_eval_batch_size = gr.Slider(label='Batch Size', value=8, minimum=1, maximum=128, step=1, info='Batch size per GPU/TPU core/CPU for evaluation.', interactive=True)
        eval_start_btn = gr.Button("Start Evaluation")
        eval_output = gr.Markdown(value="Ready")
        eval_start_btn.click(run_eval, [base_model, checkpoint, eval_dataset, per_device_eval_batch_size], eval_output)

def main():
    with open("./src/webui/css/main.css") as f:
        css = f.read()
    with gr.Blocks(css=css) as demo:
        create_sft_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)


if __name__ == "__main__":
    main()
