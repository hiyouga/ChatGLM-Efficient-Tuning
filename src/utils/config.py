import os
import json
from typing import Optional
from dataclasses import dataclass, field


CHATGLM_REPO_NAME = "THUDM/chatglm-6b"
CHATGLM_LASTEST_HASH = "a8ede826cf1b62bd3c78bdfb3625c7c5d2048fbd"


@dataclass
class DatasetAttr:

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
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: Optional[str] = field(
        default=CHATGLM_LASTEST_HASH,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Will use the token generated when running `huggingface-cli login`."}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the model checkpoints as well as the configurations."}
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the reward model."}
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
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."}
    )
    overwrite_cache: Optional[bool] = field(
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
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"}
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self): # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset.split(",")]
        dataset_info = json.load(open(os.path.join(self.dataset_dir, "dataset_info.json"), "r"))

        self.dataset_list = []
        for name in dataset_names:
            if name not in dataset_info:
                raise ValueError("Undefined dataset {} in dataset_info.json.".format(name))

            if "hf_hub_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_info[name]["hf_hub_url"])
            elif "script_url" in dataset_info[name]:
                dataset_attr = DatasetAttr("script", dataset_name=dataset_info[name]["script_url"])
            else:
                dataset_attr = DatasetAttr(
                    "file",
                    file_name=dataset_info[name]["file_name"],
                    file_sha1=dataset_info[name]["file_sha1"] if "file_sha1" in dataset_info[name] else None
                )

            if "columns" in dataset_info[name]:
                dataset_attr.prompt_column = dataset_info[name]["columns"].get("prompt", None)
                dataset_attr.query_column = dataset_info[name]["columns"].get("query", None)
                dataset_attr.response_column = dataset_info[name]["columns"].get("response", None)
                dataset_attr.history_column = dataset_info[name]["columns"].get("history", None)

            self.dataset_list.append(dataset_attr)


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[str] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for Freeze fine-tuning."}
    )
    name_module_trainable: Optional[str] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for Freeze fine-tuning."}
    )
    pre_seq_len: Optional[int] = field(
        default=16,
        metadata={"help": "Number of prefix tokens to use for P-tuning V2."}
    )
    prefix_projection: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add a project layer for the prefix in P-tuning V2 or not."}
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
        metadata={"help": "Name(s) of target modules to apply LoRA. Use comma to separate multiple modules."}
    )
    resume_lora_training: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to resume training from the last LoRA weights or create new weights after merging them."}
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to plot the training loss after fine-tuning or not."}
    )

    def __post_init__(self):
        self.lora_target = [target.strip() for target in self.lora_target.split(",")] # support custom target modules of LoRA

        if self.num_layer_trainable > 0: # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [27-k for k in range(self.num_layer_trainable)]
        else: # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-self.num_layer_trainable)]
        if self.name_module_trainable == "mlp":
            self.trainable_layers = ["layers.{:d}.mlp".format(idx) for idx in trainable_layer_ids]
        elif self.name_module_trainable == "qkv":
            self.trainable_layers = ["layers.{:d}.attention.query_key_value".format(idx) for idx in trainable_layer_ids]

        if self.finetuning_type not in ["none", "freeze", "p_tuning", "lora", "full"]:
            raise NotImplementedError("Invalid fine-tuning method.")
