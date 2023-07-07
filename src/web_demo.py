# coding=utf-8
# Implements user interface in browser for ChatGLM fine-tuned with PEFT.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/web_demo.py
# Usage: python web_demo.py --checkpoint_dir path_to_checkpoint [--quantization_bit 4]


import torch
import mdtex2html
import gradio as gr
from transformers.utils.versions import require_version

from extras.misc import auto_configure_device_map
from pet import get_infer_args, load_model_and_tokenizer


require_version("gradio>=3.30.0", "To fix: pip install gradio>=3.30.0")


model_args, finetuning_args, generating_args = get_infer_args()
model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

if torch.cuda.device_count() > 1:
    from accelerate import dispatch_model
    device_map = auto_configure_device_map(torch.cuda.device_count(), use_v2=model_args.use_v2)
    model = dispatch_model(model, device_map)
else:
    model = model.cuda()

model.eval()


"""Override Chatbot.postprocess"""

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


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


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature, do_sample=generating_args.do_sample,
                                               num_beams=generating_args.num_beams, top_k=generating_args.top_k):
        chatbot[-1] = (parse_text(input), parse_text(response))

        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:

    gr.HTML("""
    <h1 align="center">
        <a href="https://github.com/hiyouga/ChatGLM-Efficient-Tuning" target="_blank">
            ChatGLM Efficient Tuning
        </a>
    </h1>
    """)

    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=generating_args.max_length, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=generating_args.top_p, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1.5, value=generating_args.temperature, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name="0.0.0.0", share=True, inbrowser=True)
