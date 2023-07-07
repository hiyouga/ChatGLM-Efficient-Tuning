# coding=utf-8
# Implements API for ChatGLM fine-tuned with PEFT in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python api_demo.py --model_name_or_path path_to_model --checkpoint_dir path_to_checkpoint
# Visit http://localhost:8000/docs for document.


import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sse_starlette import EventSourceResponse
from typing import Any, Dict, List, Literal, Optional

from extras.misc import auto_configure_device_map
from pet import get_infer_args, load_model_and_tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: Optional[str] = "model"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    owned_by: Optional[str] = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = []


class ModelList(BaseModel):
    object: Optional[str] = "list"
    data: Optional[List[ModelCard]] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: Optional[str] = "chatcmpl-default"
    object: Literal["chat.completion"]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class ChatCompletionStreamResponse(BaseModel):
    id: Optional[str] = "chatcmpl-default"
    object: Literal["chat.completion.chunk"]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer, generating_args

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        query = prev_messages.pop(0).content + query

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i].role == "user" and prev_messages[i+1].role == "assistant":
                history.append([prev_messages[i].content, prev_messages[i+1].content])

    gen_kwargs = generating_args.to_dict()
    gen_kwargs.update({
        "temperature": request.temperature if request.temperature else gen_kwargs["temperature"],
        "top_p": request.top_p if request.top_p else gen_kwargs["top_p"]
    })

    if request.max_tokens:
        gen_kwargs.pop("max_length", None)
        gen_kwargs["max_new_tokens"] = request.max_tokens

    if request.stream:
        generate = predict(query, history, gen_kwargs, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history, **gen_kwargs)

    usage = ChatCompletionResponseUsage( # too complex to compute
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2
    )

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], usage=usage, object="chat.completion")


async def predict(query: str, history: List[List[str]], gen_kwargs: Dict[str, Any], model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionStreamResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield chunk.json(exclude_unset=True, ensure_ascii=False)

    current_length = 0

    for new_response, _ in model.stream_chat(tokenizer, query, history, **gen_kwargs):
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield chunk.json(exclude_unset=True, ensure_ascii=False)

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionStreamResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield chunk.json(exclude_unset=True, ensure_ascii=False)
    yield "[DONE]"


if __name__ == "__main__":
    model_args, finetuning_args, generating_args = get_infer_args()
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(torch.cuda.device_count(), use_v2=model_args.use_v2)
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()

    model.eval()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
