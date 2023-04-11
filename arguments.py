from typing import Optional
from dataclasses import dataclass, field
from config_data import CHATGLM_LASTEST_HASH, DATASETS


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
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
    dataset: Optional[str] = field(
        default="alpaca_zh",
        metadata={"help": "The name of provided dataset to use."}
    )
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "The name of the folder containing datasets."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and sets."}
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
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError("Undefined dataset in config_data.py.")

        if "hf_hub_url" in DATASETS[self.dataset]:
            self.load_from = "hf_hub"
            self.dataset_name = DATASETS[self.dataset]["hf_hub_url"]
        elif "script_url" in DATASETS[self.dataset]:
            self.load_from = "script"
            self.dataset_name = DATASETS[self.dataset]["script_url"]
        else:
            self.load_from = "file"
            self.train_file = DATASETS[self.dataset]["filename"]
            self.train_hash = DATASETS[self.dataset]["sha1"]

        if "columns" in DATASETS[self.dataset]:
            self.prompt_column = DATASETS[self.dataset]["columns"]["prompt"]
            self.query_column = DATASETS[self.dataset]["columns"]["query"]
            self.response_column = DATASETS[self.dataset]["columns"]["response"]
            self.history_column = DATASETS[self.dataset]["columns"]["history"]
        else:
            self.prompt_column = "instruction"
            self.query_column = "input"
            self.response_column = "output"
            self.history_column = None


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[str] = field(
        default="lora",
        metadata={"help": "The name of fine-tuning technique."}
    )
    pre_seq_len: Optional[int] = field(
        default=8,
        metadata={"help": "Number of prefix tokens to use for P-tuning v2."}
    )
    prefix_projection: bool = field(
        default=False,
        metadata={"help": "Whether to add a project layer for the prefix in P-tuning v2."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning. (similar with the learning rate)"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )

    def __post_init__(self):
        if self.finetuning_type not in ["freeze", "p_tuning", "lora"]:
            raise NotImplementedError("Invalid fine-tuning method.")
