"""Chat completion endpoints."""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from traceloop.sdk.decorators import workflow, task

from app.core.config import settings
from app.core.llm import chat_completion
from app.models.schemas import (
    ChatRequest, ChatResponse, Choice, Message, ModelsResponse, ModelInfo, Usage,
)

router = APIRouter(prefix="/chat", tags=["chat"])

AVAILABLE_MODELS: list[ModelInfo] = [
    ModelInfo(id="claude-4-6-sonnet", provider="anthropic", description="Anthropic Claude 3.5 Sonnet"),
    ModelInfo(id="claude-4-6-opus",     provider="anthropic", description="Anthropic Claude 3 Opus"),
    ModelInfo(id="claude-4-6-haiku",    provider="anthropic", description="Anthropic Claude 3 Haiku"),
    ModelInfo(id="llama3.2:latest",                   provider="ollama",    description="Meta Llama 3.2 via Ollama"),
    ModelInfo(id="mistral",                    provider="ollama",    description="Mistral 7B via Ollama"),
    ModelInfo(id="gemma2",                     provider="ollama",    description="Google Gemma 2 via Ollama"),
]


@task(name="prepare_messages")
def prepare_messages(request: ChatRequest) -> list[dict]:
    return [m.model_dump() for m in request.messages]


@task(name="parse_llm_response")
def parse_response(raw, provider: str) -> ChatResponse:
    choices = [
        Choice(index=c.index, message=Message(role=c.message.role, content=c.message.content or ""), finish_reason=c.finish_reason)
        for c in raw.choices
    ]
    usage = None
    if raw.usage:
        usage = Usage(prompt_tokens=raw.usage.prompt_tokens, completion_tokens=raw.usage.completion_tokens, total_tokens=raw.usage.total_tokens)
    return ChatResponse(id=raw.id, model=raw.model, provider=provider, choices=choices, usage=usage)


@router.post("/completions", response_model=ChatResponse, summary="Chat completion")
@workflow(name="chat_completions_workflow")
async def completions(request: ChatRequest):
    messages = prepare_messages(request)
    if request.stream:
        return await _stream_response(request, messages)
    try:
        raw = await chat_completion(messages=messages, provider=request.provider, model=request.model, temperature=request.temperature, max_tokens=request.max_tokens)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return parse_response(raw, request.provider)


async def _stream_response(request: ChatRequest, messages: list[dict]) -> StreamingResponse:
    async def event_generator():
        try:
            stream = await chat_completion(messages=messages, provider=request.provider, model=request.model, temperature=request.temperature, max_tokens=request.max_tokens, stream=True)
            async for chunk in stream:
                delta = chunk.choices[0].delta
                content = delta.content if delta and delta.content else ""
                yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/models", response_model=ModelsResponse, summary="List available models")
async def list_models():
    return ModelsResponse(models=AVAILABLE_MODELS)
