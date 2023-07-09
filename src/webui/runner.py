import transformers
import time
import os
import logging
import threading
from pet import (
    get_train_args,
    get_infer_args,
    run_sft
)
from datetime import timedelta
from webui import common

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

class Callbacks(transformers.TrainerCallback):
    def __init__(self, runner):
        self.runner = runner
        self.start_time = time.time()
        tracker_key = ["current_steps", "total_steps", "loss", "reward", "learning_rate", "epoch", "percentage", "elapsed_time", "remaining_time"]
        self.tracker = {k: None for k in tracker_key}

    def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        if self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True
    
    def on_substep_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
        if self.runner.aborted:
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

def format_info(trainer_log, tracker):
    ret_info = trainer_log
    if tracker["current_steps"]:
        ret_info += '\n'
        ret_info += f"Running... **{tracker['current_steps']} / {tracker['total_steps']}** ... {tracker['elapsed_time']} < {tracker['remaining_time']}"
    return ret_info

class Runner():
    def __init__(self):
        self.aborted = False
    
    def set_abort(self):
        self.aborted = True
    
    def run_train(self, model_name, base_model, ft_type, dataset, lr, epochs, fp16, per_device_train_batch_size, gradient_accumulation_steps, lr_scheduler_type, logging_steps, save_steps):
        self.aborted = False

        callback_handler = log_handler()
        callback_handler.setLevel(logging.INFO)
        logging.root.addHandler(callback_handler)

        save_path = os.path.join(common.web_log_dir, base_model, model_name)

        args = {
            "model_name_or_path": common.settings["path_to_model"][base_model],
            "do_train": True,
            "dataset": dataset,
            "finetuning_type": ft_type,
            "output_dir": save_path,
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

        train_callback = Callbacks(self)
        def thread_run():
            run_sft(model_args, data_args, training_args, finetuning_args, [train_callback])
        
        thread = threading.Thread(target=thread_run)
        thread.start()

        while thread.is_alive():
            time.sleep(1)
            if self.aborted:
                yield "Aborted, wait for terminating..."
            else:
                yield format_info(callback_handler.log, train_callback.tracker)
        
        yield "Ready"

    def run_eval(self, base_model, ckpt, dataset, per_device_eval_batch_size):
        save_path = os.path.join(common.web_log_dir, base_model, ckpt)
        args = {
            "model_name_or_path": common.settings["path_to_model"][base_model],
            "do_eval": True,
            "dataset": dataset,
            "checkpoint_dir": save_path,
            "output_dir": os.path.join(save_path, "eval"),
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "predict_with_generate": True
        }
        model_args, data_args, training_args, finetuning_args, general_args = get_train_args(args)
        # run_sft(model_args, data_args, training_args, finetuning_args)
        raise NotImplementedError
