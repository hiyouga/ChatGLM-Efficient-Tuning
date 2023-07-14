import os
import time
import logging
import threading
import transformers
from typing import Optional, Tuple

from glmtuner.extras.misc import torch_gc
from glmtuner.extras.callbacks import LogCallback
from glmtuner.extras.logging import LoggerHandler
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

    def initialize(self, base_model: str, model_list: list, dataset: list) -> Tuple[str, LoggerHandler, LogCallback]:
        if self.running:
            return "A process is in running, please abort it firstly.", None, None

        if not base_model:
            return "Please select a model.", None, None

        if len(model_list) == 0:
            return "No model detected.", None, None

        if len(dataset) == 0:
            return "Please choose datasets.", None, None

        self.aborted = False
        self.running = True

        logger_handler = LoggerHandler()
        logger_handler.setLevel(logging.INFO)
        logging.root.addHandler(logger_handler)
        transformers.logging.add_handler(logger_handler)
        trainer_callback = LogCallback(self)

        return "", logger_handler, trainer_callback

    def finalize(self, finish_info: Optional[str] = None) -> str:
        self.running = False
        torch_gc()
        if self.aborted:
            return "Ready"
        else:
            return finish_info if finish_info is not None else "Finished"

    def run_train(
        self, base_model, model_list, checkpoints, output_dir, finetuning_type,
        dataset, learning_rate, num_train_epochs, max_samples,
        fp16, use_v2, per_device_train_batch_size, gradient_accumulation_steps,
        lr_scheduler_type, logging_steps, save_steps
    ):
        error, logger_handler, trainer_callback = self.initialize(base_model, model_list, dataset)
        if error:
            yield error
            return

        model_path = [path for name, path in model_list if name == base_model]
        if get_save_dir(base_model) and checkpoints:
            checkpoint_dir = ",".join([os.path.join(get_save_dir(base_model), checkpoint) for checkpoint in checkpoints])
        else:
            checkpoint_dir = None

        args = dict(
            model_name_or_path=model_path[0],
            do_train=True,
            finetuning_type=finetuning_type,
            dataset=",".join(dataset),
            dataset_dir=DATA_DIR,
            max_samples=int(max_samples),
            output_dir=os.path.join(get_save_dir(base_model), output_dir),
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
            use_v2=use_v2
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
        self, base_model, model_list, checkpoints, dataset, max_samples, per_device_eval_batch_size, use_v2
    ):
        error, logger_handler, trainer_callback = self.initialize(base_model, model_list, dataset)
        if error:
            yield error
            return

        model_path = [path for name, path in model_list if name == base_model]
        if get_save_dir(base_model) and checkpoints:
            checkpoint_dir = ",".join([os.path.join(get_save_dir(base_model), checkpoint) for checkpoint in checkpoints])
            output_dir = os.path.join(get_save_dir(base_model), "eval_" + "_".join(checkpoints))
        else:
            checkpoint_dir = None
            output_dir = os.path.join(get_save_dir(base_model), "eval_base")

        args = dict(
            model_name_or_path=model_path[0],
            do_eval=True,
            dataset=",".join(dataset),
            dataset_dir=DATA_DIR,
            max_samples=int(max_samples),
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            overwrite_cache=True,
            predict_with_generate=True,
            per_device_eval_batch_size=per_device_eval_batch_size,
            use_v2=use_v2
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
