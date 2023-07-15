# coding=utf-8
# Implements stream chat in command line for ChatGLM fine-tuned with PEFT.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py
# Usage: python cli_demo.py --checkpoint_dir path_to_checkpoint [--quantization_bit 4]


import os
import torch
import signal
import platform

from glmtuner import auto_configure_device_map, get_infer_args, load_model_and_tokenizer


os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"
stop_stream = False
welcome = "欢迎使用 ChatGLM-6B 模型，输入内容即可对话，clear 清空对话历史，stop 终止程序"


def build_prompt(history):
    prompt = welcome
    for query, response in history:
        prompt += f"\n\nUser: {query}"
        prompt += f"\n\nChatGLM-6B: {response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    global stop_stream
    model_args, finetuning_args, generating_args = get_infer_args()
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(torch.cuda.device_count(), use_v2=(tokenizer.eos_token_id==2))
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()

    model.eval()

    history = []
    print(welcome)
    while True:
        try:
            query = input("\nInput: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print(welcome)
            continue

        count = 0
        for _, history in model.stream_chat(tokenizer, query, history=history, **generating_args.to_dict()):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()
