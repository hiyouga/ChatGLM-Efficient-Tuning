# Inspired by:
# https://github.com/lvwerra/trl/blob/main/examples/summarization/scripts/reward_summarization.py
# https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py


from transformers import Seq2SeqTrainingArguments

from glmtuner.dsets import get_dataset, preprocess_dataset
from glmtuner.extras.callbacks import LogCallback
from glmtuner.extras.ploting import plot_loss
from glmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments
from glmtuner.tuner.core import load_model_and_tokenizer
from glmtuner.tuner.rm.metric import compute_accuracy
from glmtuner.tuner.rm.collator import PairwiseDataCollatorForChatGLM
from glmtuner.tuner.rm.trainer import PairwiseTrainerForChatGLM


def run_rm(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="rm")
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="rm")
    data_collator = PairwiseDataCollatorForChatGLM(tokenizer, model.pretrained_model, use_v2=model_args.use_v2)

    training_args.remove_unused_columns = False # Important for pairwise dataset

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
    trainer = PairwiseTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        compute_metrics=compute_accuracy,
        **trainer_kwargs
    )

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
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
