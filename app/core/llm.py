"""LiteLLM wrapper with provider resolution for Anthropic and Ollama."""

import litellm
from typing import AsyncIterator
from app.core.config import settings

litellm.set_verbose = settings.debug
litellm.ollama_key = "ollama"


def _resolve_model(provider: str, model: str | None) -> str:
    if provider == "anthropic":
        base = model or settings.anthropic_default_model
        return base if base.startswith("anthropic/") else f"anthropic/{base}"
    if provider == "ollama":
        base = model or settings.ollama_default_model
        return base if base.startswith("ollama/") else f"ollama/{base}"
    raise ValueError(f"Unsupported provider: {provider!r}. Choose 'anthropic' or 'ollama'.")


def _extra_kwargs(provider: str) -> dict:
    if provider == "ollama":
        return {"api_base": settings.ollama_base_url}
    if provider == "anthropic":
        return {"api_key": settings.anthropic_api_key} if settings.anthropic_api_key else {}
    return {}


async def chat_completion(
    messages: list[dict],
    provider: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
) -> dict | AsyncIterator:
    resolved_model = _resolve_model(provider, model)
    extra = _extra_kwargs(provider)
    response = await litellm.acompletion(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        **extra,
    )
    return response
