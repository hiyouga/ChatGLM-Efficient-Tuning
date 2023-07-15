import logging
import os
import threading
import time
import transformers
from typing import Optional, Tuple

from glmtuner.extras.callbacks import LogCallback
from glmtuner.extras.constants import SUPPORTED_MODELS
from glmtuner.extras.logging import LoggerHandler
from glmtuner.extras.misc import torch_gc
from glmtuner.tuner import get_train_args, run_sft
from glmtuner.webui.common import get_save_dir, DATA_DIR
from glmtuner.webui.utils import format_info, get_eval_results


class Runner:

    def __init__(self):
        self.aborted = False
        self.running = False

    def set_abort(self):
        self.aborted = True
        self.running = False

    def initialize(self, model_name: str, model_path: str, dataset: list) -> Tuple[str, str, LoggerHandler, LogCallback]:
        if self.running:
            return None, "A process is in running, please abort it firstly.", None, None

        if not model_name:
            return None, "Please select a model.", None, None

        if model_path:
            if not os.path.isdir(model_path):
                return None, "Cannot find model directory in local disk.", None, None
            model_name_or_path = model_path
        elif model_name in SUPPORTED_MODELS: # TODO: use list in gr.State
            model_name_or_path = SUPPORTED_MODELS[model_name]["hf_path"]
        else:
            return None, "Invalid model.", None, None

        if len(dataset) == 0:
            return None, "Please choose datasets.", None, None

        self.aborted = False
        self.running = True

        logger_handler = LoggerHandler()
        logger_handler.setLevel(logging.INFO)
        logging.root.addHandler(logger_handler)
        transformers.logging.add_handler(logger_handler)
        trainer_callback = LogCallback(self)

        return model_name_or_path, "", logger_handler, trainer_callback

    def finalize(self, finish_info: Optional[str] = None) -> str:
        self.running = False
        torch_gc()
        if self.aborted:
            return "Ready"
        else:
            return finish_info if finish_info is not None else "Finished"

    def run_train(
        self, model_name, model_path, checkpoints, output_dir, finetuning_type,
        dataset, learning_rate, num_train_epochs, max_samples,
        fp16, quantization_bit, per_device_train_batch_size, gradient_accumulation_steps,
        lr_scheduler_type, logging_steps, save_steps
    ):
        model_name_or_path, error, logger_handler, trainer_callback = self.initialize(model_name, model_path, dataset)
        if error:
            yield error
            return

        if checkpoints:
            checkpoint_dir = ",".join([os.path.join(get_save_dir(model_name), checkpoint) for checkpoint in checkpoints])
        else:
            checkpoint_dir = None

        args = dict(
            model_name_or_path=model_name_or_path,
            do_train=True,
            finetuning_type=finetuning_type,
            dataset=",".join(dataset),
            dataset_dir=DATA_DIR,
            max_samples=int(max_samples),
            output_dir=os.path.join(get_save_dir(model_name), output_dir),
            checkpoint_dir=checkpoint_dir,
            overwrite_cache=True,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            save_steps=save_steps,
            learning_rate=float(learning_rate),
            num_train_epochs=float(num_train_epochs),
            fp16=fp16,
            quantization_bit=int(quantization_bit) if quantization_bit else None
        )
        model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)

        run_args = dict(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            callbacks=[trainer_callback]
        )
        thread = threading.Thread(target=run_sft, kwargs=run_args)
        thread.start()

        while thread.is_alive():
            time.sleep(1)
            if self.aborted:
                yield "Aborted, wait for terminating..."
            else:
                yield format_info(logger_handler.log, trainer_callback.tracker)

        yield self.finalize()

    def run_eval(
        self, model_name, model_path, checkpoints, dataset, max_samples, per_device_eval_batch_size,
        quantization_bit
    ):
        model_name_or_path, error, logger_handler, trainer_callback = self.initialize(model_name, model_path, dataset)
        if error:
            yield error
            return

        if checkpoints:
            checkpoint_dir = ",".join([os.path.join(get_save_dir(model_name), checkpoint) for checkpoint in checkpoints])
            output_dir = os.path.join(get_save_dir(model_name), "eval_" + "_".join(checkpoints))
        else:
            checkpoint_dir = None
            output_dir = os.path.join(get_save_dir(model_name), "eval_base")

        args = dict(
            model_name_or_path=model_name_or_path,
            do_eval=True,
            dataset=",".join(dataset),
            dataset_dir=DATA_DIR,
            max_samples=int(max_samples),
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            overwrite_cache=True,
            predict_with_generate=True,
            per_device_eval_batch_size=per_device_eval_batch_size,
            quantization_bit=int(quantization_bit) if quantization_bit else None
        )
        model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)

        run_args = dict(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            callbacks=[trainer_callback]
        )
        thread = threading.Thread(target=run_sft, kwargs=run_args)
        thread.start()

        while thread.is_alive():
            time.sleep(1)
            if self.aborted:
                yield "Aborted, wait for terminating..."
            else:
                yield format_info(logger_handler.log, trainer_callback.tracker)

        yield self.finalize(get_eval_results(os.path.join(output_dir, "eval_results.json")))
