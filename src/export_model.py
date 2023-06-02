# coding=utf-8
# Exports the fine-tuned ChatGLM-6B model.
# Usage: python export_model.py --checkpoint_dir path_to_checkpoint --output_dir path_to_save_model


from utils import ModelArguments, FinetuningArguments, load_pretrained
from transformers import HfArgumentParser, TrainingArguments
from transformers.utils.versions import require_version


def main():

    require_version("transformers==4.27.4", "To fix: pip install transformers==4.27.4") # higher version may cause problems

    parser = HfArgumentParser((ModelArguments, TrainingArguments, FinetuningArguments))
    model_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_pretrained(model_args, finetuning_args)
    model.save_pretrained(training_args.output_dir, max_shard_size="1GB")
    tokenizer.save_pretrained(training_args.output_dir)

    print("model and tokenizer have been saved at:", training_args.output_dir)


if __name__ == "__main__":
    main()
