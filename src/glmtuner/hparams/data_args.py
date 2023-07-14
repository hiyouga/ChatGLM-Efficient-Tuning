import os
import json
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class DatasetAttr:

    load_from: str
    dataset_name: str
    dataset_sha1: Optional[str] = None

    def __repr__(self) -> str:
        return self.dataset_name

    def __post_init__(self):
        self.prompt_column = "instruction"
        self.query_column = "input"
        self.response_column = "output"
        self.history_column = None


@dataclass
class DataArguments:
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
    eval_num_beams: Optional[int] = field(
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
    dev_ratio: Optional[float] = field(
        default=0,
        metadata={"help": "Proportion of the dataset to include in the development set, should be between 0.0 and 1.0."}
    )

    def init_for_training(self): # support mixing multiple datasets
        dataset_names = [ds.strip() for ds in self.dataset.split(",")]
        with open(os.path.join(self.dataset_dir, "dataset_info.json"), "r", encoding="utf-8") as f:
            dataset_info = json.load(f)

        self.dataset_list: List[DatasetAttr] = []
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
                    dataset_name=dataset_info[name]["file_name"],
                    dataset_sha1=dataset_info[name].get("file_sha1", None)
                )

            if "columns" in dataset_info[name]:
                dataset_attr.prompt_column = dataset_info[name]["columns"].get("prompt", None)
                dataset_attr.query_column = dataset_info[name]["columns"].get("query", None)
                dataset_attr.response_column = dataset_info[name]["columns"].get("response", None)
                dataset_attr.history_column = dataset_info[name]["columns"].get("history", None)

            self.dataset_list.append(dataset_attr)
