"""FastAPI application entry point with TraceLoop instrumentation."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from traceloop.sdk import Traceloop

from app.core.config import settings
from app.models.schemas import HealthResponse
from app.routers import chat

# Traceloop.init(
#     app_name="llm_chatbot",
#     api_endpoint="http://localhost:4318",
#     disable_batch=True,
# )


def _init_traceloop() -> None:
    print("Initializing traceloop...")
    init_kwargs: dict = {
        "app_name": settings.app_name,
        "disable_batch": settings.debug,
        "api_endpoint": settings.traceloop_base_url
    }

    if settings.traceloop_api_key:
        init_kwargs["api_key"] = settings.traceloop_api_key
    if settings.traceloop_base_url:
        init_kwargs["api_endpoint"] = settings.traceloop_base_url
    Traceloop.init(**init_kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_traceloop()
    yield


app = FastAPI(
    title="FastAPI LLM Gateway",
    description="FastAPI + LiteLLM gateway for Anthropic and Ollama, traced with TraceLoop.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(chat.router)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(status="ok", app=settings.app_name)


@app.get("/", tags=["ops"])
async def root():
    return {"message": "FastAPI LLM Gateway is running", "docs": "/docs", "health": "/health"}
