import torch
from typing import Any, Dict, Generator, List, Optional, Tuple

from glmtuner.extras.misc import auto_configure_device_map
from glmtuner.hparams import ModelArguments, FinetuningArguments, GeneratingArguments
from glmtuner.tuner import load_model_and_tokenizer


class ChatModel:

    def __init__(
        self,
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        generating_args: GeneratingArguments
    ) -> None:
        self.model, self.tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

        if torch.cuda.device_count() > 1:
            from accelerate import dispatch_model
            device_map = auto_configure_device_map(torch.cuda.device_count(), use_v2=(self.tokenizer.eos_token_id==2))
            self.model = dispatch_model(self.model, device_map)
        else:
            self.model = self.model.cuda()

        self.model.eval()
        self.generating_args = generating_args

    def process_args(self, **input_kwargs) -> Dict[str, Any]:
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)

        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs.update(dict(
            temperature=temperature or gen_kwargs["temperature"],
            top_p=top_p or gen_kwargs["top_p"],
            top_k=top_k or gen_kwargs["top_k"],
            repetition_penalty=repetition_penalty or gen_kwargs["repetition_penalty"]
        ))

        if max_length:
            gen_kwargs.pop("max_new_tokens", None)
            gen_kwargs["max_length"] = max_length

        if max_new_tokens:
            gen_kwargs.pop("max_length", None)
            gen_kwargs["max_new_tokens"] = max_new_tokens

        return gen_kwargs

    def chat(self, query: str, history: Optional[List[Tuple[str, str]]] = None, **input_kwargs) -> str:
        gen_kwargs = self.process_args(**input_kwargs)
        response = self.model.chat(self.tokenizer, query, history, **gen_kwargs)
        return response

    def stream_chat(
        self, query: str, history: Optional[List[Tuple[str, str]]] = None, **input_kwargs
    ) -> Generator[str, None, None]:
        gen_kwargs = self.process_args(**input_kwargs)
        current_length = 0
        for new_response, _ in self.model.stream_chat(self.tokenizer, query, history, **gen_kwargs):
            if len(new_response) == current_length:
                continue

            new_text = new_response[current_length:]
            current_length = len(new_response)
            yield new_text
