from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    provider: Literal["anthropic", "ollama"] = "anthropic"
    model: str | None = Field(None, examples=["claude-3-5-sonnet-20241022", "llama3.2"])
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)
    stream: bool = False


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None


class ChatResponse(BaseModel):
    id: str
    model: str
    provider: str
    choices: list[Choice]
    usage: Usage | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str


class ModelInfo(BaseModel):
    id: str
    provider: str
    description: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
