# coding=utf-8
# Implements several parameter-efficient supervised fine-tuning method for ChatGLM.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py


from utils import (
    load_pretrained,
    prepare_args,
    prepare_data,
    preprocess_data,
    plot_loss,
    Seq2SeqDataCollatorForChatGLM,
    ComputeMetrics,
    Seq2SeqTrainerForChatGLM
)


def main():

    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args()
    dataset = prepare_data(model_args, data_args)
    model, tokenizer = load_pretrained(model_args, training_args, finetuning_args, training_args.do_train, stage="sft")
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="sft")
    data_collator = Seq2SeqDataCollatorForChatGLM(
        tokenizer=tokenizer,
        model=model,
        ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
        inference_mode=(not training_args.do_train)
    )

    # Override the decoding parameters of Trainer
    training_args.generation_max_length = training_args.generation_max_length if \
                training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.num_beams if \
                data_args.num_beams is not None else training_args.generation_num_beams

    # Initialize our Trainer
    trainer = Seq2SeqTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        train_dataset=dataset if training_args.do_train else None,
        eval_dataset=dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = {
        "do_sample": True,
        "top_p": 0.7,
        "max_length": 768,
        "temperature": 0.95
    }

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state() # along with the loss values
        trainer.save_model()
        if finetuning_args.plot_loss:
            plot_loss(training_args)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results, tokenizer)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
