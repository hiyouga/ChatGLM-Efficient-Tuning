#!/usr/bin/env python
# coding=utf-8
# Implement several parameter-efficient fine-tuning method for ChatGLM.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py


from utils import (
    load_pretrained,
    prepare_args,
    prepare_data,
    preprocess_data,
    plot_loss,
    DataCollatorForChatGLM,
    ComputeMetrics,
    TrainerForChatGLM
)


def main():

    # Prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args()
    dataset = prepare_data(model_args, data_args, training_args)
    model, tokenizer = load_pretrained(model_args, finetuning_args, is_trainable=training_args.do_train)
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args)
    data_collator = DataCollatorForChatGLM(tokenizer, model, data_args.ignore_pad_token_for_loss, training_args.do_eval)

    # Override the decoding parameters of Trainer
    training_args.generation_max_length = training_args.generation_max_length if \
                training_args.generation_max_length is not None else data_args.max_target_length
    training_args.generation_num_beams = data_args.num_beams if \
                data_args.num_beams is not None else training_args.generation_num_beams

    # Initialize our Trainer
    trainer = TrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        train_dataset=dataset if training_args.do_train else None,
        eval_dataset=dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None
    )

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
        model = model.half() # don't use `--fp16` argument at evaluation
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=768, temperature=0.95)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
