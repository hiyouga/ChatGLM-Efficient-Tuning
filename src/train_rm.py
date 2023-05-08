# coding=utf-8
# Implements parameter-efficient training of a reward model based on ChatGLM.
# This code is inspired by:
# https://github.com/lvwerra/trl/blob/main/examples/summarization/scripts/reward_summarization.py
# https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/train_reward_model_gptj.py


from utils import (
    prepare_args,
    prepare_data,
    load_pretrained,
    preprocess_data,
    PairwiseDataCollatorForChatGLM,
    PairwiseTrainerForChatGLM,
    plot_loss
)

def main():

    # prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args()
    dataset = prepare_data(model_args, data_args)
    model, tokenizer = load_pretrained(model_args, training_args, finetuning_args, training_args.do_train, stage="rwd")
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="rwd")
    data_collator = PairwiseDataCollatorForChatGLM(
        tokenizer=tokenizer,
        inference_mode=(not training_args.do_train)
    )

    # Initialize our Trainer
    trainer = PairwiseTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        train_dataset=dataset if training_args.do_train else None,
        eval_dataset=dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
