# coding=utf-8
# Exports the fine-tuned ChatGLM-6B model.
# Usage: python export_model.py --checkpoint_dir path_to_checkpoint --output_dir path_to_save_model


from pet import get_train_args, load_model_and_tokenizer


def main():

    model_args, _, training_args, finetuning_args, _ = get_train_args()
    model_args.use_fast_tokenizer = False
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    model.save_pretrained(training_args.output_dir, max_shard_size="1GB")
    tokenizer.save_pretrained(training_args.output_dir)
    print("model and tokenizer have been saved at:", training_args.output_dir)


if __name__ == "__main__":
    main()
