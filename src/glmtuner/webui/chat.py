import os
import torch

from glmtuner.extras.misc import auto_configure_device_map, torch_gc
from glmtuner.hparams import GeneratingArguments
from glmtuner.tuner import get_infer_args, load_model_and_tokenizer
from glmtuner.webui.common import get_save_dir


class ChatModel:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.gen_args = GeneratingArguments()

    def load_model(self, base_model: str, model_list: list, checkpoints: list, use_v2: bool):
        if self.model is not None:
            yield "You have loaded a model, please unload it first."
            return

        if not base_model:
            yield "Please select a model."
            return

        if len(model_list) == 0:
            yield "No model detected."
            return

        model_path = [path for name, path in model_list if name == base_model]
        if get_save_dir(base_model) and checkpoints:
            checkpoint_dir = ",".join([os.path.join(get_save_dir(base_model), checkpoint) for checkpoint in checkpoints])
        else:
            checkpoint_dir = None

        args = dict(
            model_name_or_path=model_path[0],
            checkpoint_dir=checkpoint_dir
        )
        model_args, finetuning_args, self.gen_args = get_infer_args(args)

        yield "Loading model..."
        self.model, self.tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

        if torch.cuda.device_count() > 1:
            from accelerate import dispatch_model
            device_map = auto_configure_device_map(torch.cuda.device_count(), use_v2=(self.tokenizer.eos_token_id==2))
            self.model = dispatch_model(self.model, device_map)
        else:
            self.model = self.model.cuda()

        self.model.eval()
        yield "Model loaded, now you can chat with your model."

    def unload_model(self):
        yield "Unloading model..."
        self.model = None
        self.tokenizer = None
        torch_gc()
        yield "Model unloaded, please load a model first."

    def predict(self, chatbot, query, history, max_length, top_p, temperature):
        chatbot.append([query, ""])
        for response, history in self.model.stream_chat(
            self.tokenizer, query, history, max_length=max_length, top_p=top_p,
            temperature=temperature, do_sample=self.gen_args.do_sample,
            num_beams=self.gen_args.num_beams, top_k=self.gen_args.top_k
        ):
            chatbot[-1] = [query, response]
            yield chatbot, history
