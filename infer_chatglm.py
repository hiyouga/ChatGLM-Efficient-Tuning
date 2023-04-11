#!/usr/bin/env python
# coding=utf-8
# Implement stream chat in command line for ChatGLM finetuned with PEFT.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py


import os
import signal
import platform
from arguments import CHATGLM_LASTEST_HASH
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModel


os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"
stop_stream = False
welcome = "欢迎使用 ChatGLM-6B 模型，输入内容即可对话，clear清空对话历史，stop终止程序"


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
    config = PeftConfig.from_pretrained("output")
    model = AutoModel.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, revision=CHATGLM_LASTEST_HASH)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, revision=CHATGLM_LASTEST_HASH)
    model = PeftModel.from_pretrained(model, "output")
    model = model.half().cuda()
    history = []
    global stop_stream
    print(welcome)
    while True:
        query = input("\nInput:")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print(welcome)
            continue
        count = 0
        for _, history in model.stream_chat(tokenizer, query, history=history):
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
