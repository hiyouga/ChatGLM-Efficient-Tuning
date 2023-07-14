import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sse_starlette import EventSourceResponse
from typing import Any, Dict, List

from glmtuner.extras.misc import auto_configure_device_map
from glmtuner.tuner import get_infer_args, load_model_and_tokenizer
from glmtuner.extras.misc import torch_gc
from glmtuner.api.protocol import (
    ModelCard,
    ModelList,
    ChatMessage,
    DeltaMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionResponseUsage
)


@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
    yield
    torch_gc()


def create_app():
    model_args, finetuning_args, generating_args = get_infer_args()
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(torch.cuda.device_count(), use_v2=model_args.use_v2)
        model = dispatch_model(model, device_map)
    else:
        model = model.cuda()

    model.eval()

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        model_card = ModelCard(id="gpt-3.5-turbo")
        return ModelList(data=[model_card])

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
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

    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
