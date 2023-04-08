from typing import Optional
from dataclasses import dataclass, field


CHATGLM_LASTEST_HASH = 'acd41f77311be8584836edc2fc7251d5b6e65840'
DATASETS = {
    "alpaca_en": {
        "train": {
            "filename": "alpaca_data_en_52k.json",
            "sha1": "607f94a7f581341e59685aef32f531095232cf23"
        }
    },
    "alpaca_zh": {
        "train": {
            "filename": "alpaca_data_zh_51k.json",
            "sha1": "e655af3db557a4197f7b0cf92e1986b08fae6311"
        },
        "val": { # for debugging
            "filename": "alpaca_data_zh_51k.json",
            "sha1": "e655af3db557a4197f7b0cf92e1986b08fae6311"
        }
    }
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to finetune.
    """
    model_name_or_path: Optional[str] = field(
        default="THUDM/chatglm-6b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: str = field(
        default=CHATGLM_LASTEST_HASH,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `huggingface-cli login`."}
    )
    resize_position_embeddings: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to resize the position embeddings if `max_source_length` exceeds."}
    )
    quantization_bit: Optional[int] = field(
        default=None
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    lang: Optional[str] = field(
        default=None,
        metadata={"help": "Language id for summarization."}
    )
    dataset_name: Optional[str] = field(
        default="alpaca_zh",
        metadata={"help": "The name of custom dataset to use."}
    )
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "The name of the folder containing datasets."}
    )
    prompt_column: Optional[str] = field(
        default="instruction",
        metadata={"help": "The name of the column in the datasets containing the prompts."}
    )
    query_column: Optional[str] = field(
        default="input",
        metadata={"help": "The name of the column in the datasets containing the querys."}
    )
    response_column: Optional[str] = field(
        default="output",
        metadata={"help": "The name of the column in the datasets containing the responses."}
    )
    history_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the history of chat."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum total output sequence length after tokenization."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Whether to pad all samples to model maximum sentence length or not."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name not in DATASETS:
            raise NotImplementedError("Invalid dataset name.")
        self.train_file = DATASETS[self.dataset_name]["train"]["filename"]
        self.train_hash = DATASETS[self.dataset_name]["train"]["sha1"]
        self.validation_file = DATASETS[self.dataset_name]["val"]["filename"] if "val" in DATASETS[self.dataset_name] else None
        self.validation_hash = DATASETS[self.dataset_name]["val"]["sha1"] if "val" in DATASETS[self.dataset_name] else None


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to which techniques we are going to finetuning with.
    """
    finetuning_type: Optional[str] = field(
        default="lora",
        metadata={"help": "The name of finetuning technique."}
    )
    pre_seq_len: Optional[int] = field(
        default=None,
        metadata={"help": "Number of tokens to use for p-tuning."}
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to add a project layer for the prefix in p-tuning."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA finetuning."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA finetuning. (similar with the learning rate)"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA finetuning."}
    )

    def __post_init__(self):
        if self.finetuning_type not in ["freeze", "p-tuning", "lora"]:
            raise NotImplementedError("Invalid finetuning technique.")
