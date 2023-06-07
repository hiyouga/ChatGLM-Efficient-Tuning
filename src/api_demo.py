# coding=utf-8
# Implements API for ChatGLM fine-tuned with PEFT.
# This code is largely borrowed from https://github.com/THUDM/ChatGLM-6B/blob/main/api.py
# Usage: python api_demo.py --checkpoint_dir path_to_checkpoint [--quantization_bit 4]

# Request:
# curl http://127.0.0.1:8000 --header 'Content-Type: application/json' --data '{"prompt": "Hello there!", "history": []}'

# Response:
# {
#   "response": "'Hi there!'",
#   "history": "[('Hello there!', 'Hi there!')]",
#   "status": 200,
#   "time": "2000-00-00 00:00:00"
# }


import json
import torch
import uvicorn
import datetime
from fastapi import FastAPI, Request

from utils import ModelArguments, FinetuningArguments, auto_configure_device_map, load_pretrained
from transformers import HfArgumentParser


def torch_gc():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for device_id in range(num_gpus):
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer

    # Parse the request JSON
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get("prompt")
    history = json_post_list.get("history")

    # Generate response
    response, history = model.chat(tokenizer, prompt, history=history)

    # Prepare response
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": repr(response),
        "history": repr(history),
        "status": 200,
        "time": time
    }

    # Log and clean up
    log = "[" + time + "] " + "\", prompt:\"" + prompt + "\", response:\"" + repr(response) + "\""
    print(log)
    torch_gc()

    return answer


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, FinetuningArguments))
    model_args, finetuning_args = parser.parse_args_into_dataclasses()
    model, tokenizer = load_pretrained(model_args, finetuning_args)

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(torch.cuda.device_count())
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()

    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
