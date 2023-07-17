import os
from typing import List, Tuple

from glmtuner.chat.stream_chat import ChatModel
from glmtuner.extras.constants import SUPPORTED_MODELS
from glmtuner.extras.misc import torch_gc
from glmtuner.hparams import GeneratingArguments
from glmtuner.tuner import get_infer_args
from glmtuner.webui.common import get_save_dir


class WebChatModel(ChatModel):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generating_args = GeneratingArguments()

    def load_model(
        self, lang: str, model_name: str, model_path: str, checkpoints: list,
        finetuning_type: str, quantization_bit: str
    ):
        if self.model is not None:
            yield "You have loaded a model, please unload it first."
            return

        if not model_name:
            yield "Please select a model."
            return

        if model_path:
            if not os.path.isdir(model_path):
                return None, "Cannot find model directory in local disk.", None, None
            model_name_or_path = model_path
        elif model_name in SUPPORTED_MODELS: # TODO: use list in gr.State
            model_name_or_path = SUPPORTED_MODELS[model_name]["hf_path"]
        else:
            return None, "Invalid model.", None, None

        if checkpoints:
            checkpoint_dir = ",".join([os.path.join(get_save_dir(model_name), checkpoint) for checkpoint in checkpoints])
        else:
            checkpoint_dir = None

        yield "Loading model..."
        args = dict(
            model_name_or_path=model_name_or_path,
            finetuning_type=finetuning_type,
            checkpoint_dir=checkpoint_dir,
            quantization_bit=int(quantization_bit) if quantization_bit else None
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
