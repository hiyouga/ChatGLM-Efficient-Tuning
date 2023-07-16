import json
from typing import Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class FinetuningArguments:
    """
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    finetuning_type: Optional[Literal["none", "freeze", "p_tuning", "lora", "full"]] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for Freeze fine-tuning."}
    )
    name_module_trainable: Optional[Literal["mlp", "qkv"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for Freeze fine-tuning."}
    )
    pre_seq_len: Optional[int] = field(
        default=64,
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
        metadata={"help": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules."}
    )

    def __post_init__(self):
        if isinstance(self.lora_target, str):
            self.lora_target = [target.strip() for target in self.lora_target.split(",")] # support custom target modules of LoRA

        if self.num_layer_trainable > 0: # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [27 - k for k in range(self.num_layer_trainable)]
        else: # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-self.num_layer_trainable)]

        if self.name_module_trainable == "mlp":
            self.trainable_layers = ["{:d}.mlp".format(idx) for idx in trainable_layer_ids]
        elif self.name_module_trainable == "qkv":
            self.trainable_layers = ["{:d}.attention.query_key_value".format(idx) for idx in trainable_layer_ids]

        assert self.finetuning_type in ["none", "freeze", "p_tuning", "lora", "full"], "Invalid fine-tuning method."

    def save_to_json(self, json_path: str) -> None:
        """Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
