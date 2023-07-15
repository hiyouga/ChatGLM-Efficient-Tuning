# Inspired by: https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py

from typing import Optional, List
from transformers import Seq2SeqTrainingArguments, TrainerCallback

from glmtuner.dsets import DataCollatorForChatGLM, get_dataset, preprocess_dataset
from glmtuner.extras.callbacks import LogCallback
from glmtuner.extras.misc import get_logits_processor
from glmtuner.extras.ploting import plot_loss
from glmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments
from glmtuner.tuner.core import load_model_and_tokenizer
from glmtuner.tuner.sft.metric import ComputeMetrics
from glmtuner.tuner.sft.trainer import Seq2SeqTrainerForChatGLM


def run_sft(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    callbacks: Optional[List[TrainerCallback]] = [LogCallback()]
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")
    data_collator = DataCollatorForChatGLM(
        tokenizer=tokenizer,
        model=model,
        ignore_pad_token_for_loss=(data_args.ignore_pad_token_for_loss and not training_args.predict_with_generate)
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length if \
                training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.eval_num_beams if \
                data_args.eval_num_beams is not None else training_args.generation_num_beams

    # Split the dataset
    if training_args.do_train:
        if data_args.dev_ratio > 1e-6:
            dataset = dataset.train_test_split(test_size=data_args.dev_ratio)
            trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            trainer_kwargs = {"train_dataset": dataset}
    else: # do_eval or do_predict
        trainer_kwargs = {"eval_dataset": dataset}

    # Initialize our Trainer
    trainer = Seq2SeqTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **trainer_kwargs
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = {
        "do_sample": True,
        "top_p": 0.7,
        "max_new_tokens": data_args.max_target_length + 1,
        "temperature": 0.95,
        "logits_processor": get_logits_processor()
    }

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)
