from typing import Optional
from dataclasses import dataclass, field
from config_data import CHATGLM_REPO_NAME, CHATGLM_LASTEST_HASH, DATASETS


@dataclass
class DatasetInfo:

    load_from: str
    dataset_name: Optional[str] = None
    file_name: Optional[str] = None
    file_sha1: Optional[str] = None

    def __post_init__(self):
        self.prompt_column = "instruction"
        self.query_column = "input"
        self.response_column = "output"
        self.history_column = None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: Optional[str] = field(
        default=CHATGLM_REPO_NAME,
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
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the model checkpoints as well as the configurations."}
    )

    def __post_init__(self):
        if self.checkpoint_dir is not None: # support merging lora weights
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: Optional[str] = field(
        default="alpaca_zh",
        metadata={"help": "The name of provided dataset(s) to use. Use comma to separate multiple datasets."}
    )
    dataset_dir: Optional[str] = field(
        default="data",
        metadata={"help": "The name of the folder containing datasets."}
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
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total output sequence length after tokenization."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Whether to pad all samples to model maximum sentence length or not."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples for each dataset."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples for each dataset."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``"}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self): # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset.split(",")]

        self.dataset_list = []
        for name in dataset_names:
            if name not in DATASETS:
                raise ValueError("Undefined dataset {} in config_data.py.".format(name))

            if "hf_hub_url" in DATASETS[name]:
                dataset_info = DatasetInfo("hf_hub", dataset_name=DATASETS[name]["hf_hub_url"])
            elif "script_url" in DATASETS[name]:
                dataset_info = DatasetInfo("script", dataset_name=DATASETS[name]["script_url"])
            else:
                dataset_info = DatasetInfo(
                    "file",
                    file_name=DATASETS[name]["file_name"],
                    file_sha1=DATASETS[name]["file_sha1"] if "file_sha1" in DATASETS[name] else None
                )

            if "columns" in DATASETS[name]:
                dataset_info.prompt_column = DATASETS[name]["columns"]["prompt"]
                dataset_info.query_column = DATASETS[name]["columns"]["query"]
                dataset_info.response_column = DATASETS[name]["columns"]["response"]
                dataset_info.history_column = DATASETS[name]["columns"]["history"]

            self.dataset_list.append(dataset_info)


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[str] = field(
        default="lora",
        metadata={"help": "The name of fine-tuning technique."}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for Freeze fine-tuning."}
    )
    pre_seq_len: Optional[int] = field(
        default=16,
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
    lora_target: Optional[str] = field(
        default="query_key_value",
        metadata={"help": "The name(s) of target modules to apply LoRA. Use comma to separate multiple modules."}
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )

    def __post_init__(self):
        self.lora_target = [target.strip() for target in self.lora_target.split(",")] # support custom target modules of LoRA

        if self.finetuning_type not in ["none", "freeze", "p_tuning", "lora"]:
            raise NotImplementedError("Invalid fine-tuning method.")


@dataclass
class UtilArguments:
    """
    Arguments pertaining to the utilities.
    """
    do_plot: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable the plot function."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the model checkpoints as well as the configurations."}
    )
