import os
from typing import List, Tuple

from glmtuner.chat.stream_chat import ChatModel
from glmtuner.extras.misc import torch_gc
from glmtuner.hparams import GeneratingArguments
from glmtuner.tuner import get_infer_args
from glmtuner.webui.common import get_save_dir


class WebChatModel(ChatModel):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generating_args = GeneratingArguments()

    def load_model(self, base_model: str, model_path: str, checkpoints: list, quantization_bit: str):
        if self.model is not None:
            yield "You have loaded a model, please unload it first."
            return

        if not base_model:
            yield "Please select a model."
            return

        if get_save_dir(base_model) and checkpoints:
            checkpoint_dir = ",".join(
                [os.path.join(get_save_dir(base_model), checkpoint) for checkpoint in checkpoints])
        else:
            checkpoint_dir = None

        yield "Loading model..."
        if model_path:
            model_name_or_path = model_path
        else:
            model_name_or_path = base_model
        args = dict(
            model_name_or_path=model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            quantization_bit=quantization_bit
        )
        super().__init__(*get_infer_args(args))

        yield "Model loaded, now you can chat with your model."

    def unload_model(self):
        yield "Unloading model..."
        self.model = None
        self.tokenizer = None
        torch_gc()
        yield "Model unloaded, please load a model first."

    def predict(
            self,
            chatbot: List[Tuple[str, str]],
            query: str,
            history: List[Tuple[str, str]],
            max_length: int,
            top_p: float,
            temperature: float
    ):
        chatbot.append([query, ""])
        response = ""
        for new_text in self.stream_chat(query, history, max_length=max_length, top_p=top_p, temperature=temperature):
            response += new_text
            new_history = history + [(query, response)]
            chatbot[-1] = [query, response]
            yield chatbot, new_history
