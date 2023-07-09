import gc
import torch
import mdtex2html
from webui import common
from extras.misc import auto_configure_device_map
from pet import get_infer_args, load_model_and_tokenizer


def clear_torch_cache():
    gc.collect()
    if common.device == 'cuda':
        torch.cuda.empty_cache()

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

def parse_text(text): # copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

class Chater():
    def __init__(self):
        self.model = None
        self.tokenizer = None
        _, _, self.generating_args = get_infer_args({})

    def load_model(self, model_name, ckpt=None):
        if not ckpt:
            ckpt = None
        args = {
            'model_name_or_path': common.settings["path_to_model"][model_name],
            'checkpoint_dir': ckpt
        }
        model_args, finetuning_args, self.generating_args = get_infer_args(args)
        yield "Loading model..."
        self.model, self.tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
        if torch.cuda.device_count() > 1:
            from accelerate import dispatch_model
            device_map = auto_configure_device_map(torch.cuda.device_count(), use_v2=model_args.use_v2)
            self.model = dispatch_model(self.model, device_map)
        else:
            self.model = self.model.cuda()

        self.model.eval()
        yield "Model loaded, now you can use your model to infer."

    def unload_model(self):
        yield "Unloading model..."
        self.model = None
        self.tokenizer = None
        
        clear_torch_cache()
        yield 'Model unloaded, please load a model first.'
    
    def predict(self, input, chatbot, max_length, top_p, temperature, history):
        chatbot.append((parse_text(input), ""))
        for response, history in self.model.stream_chat(self.tokenizer, input, history, max_length=max_length, top_p=top_p,
                                                temperature=temperature, do_sample=self.generating_args.do_sample,
                                                num_beams=self.generating_args.num_beams, top_k=self.generating_args.top_k):
            chatbot[-1] = (parse_text(input), parse_text(response))

            yield chatbot, history
