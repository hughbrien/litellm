import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "fastapi-llm"
    debug: bool = False


    anthropic_api_key: str =os.getenv("ANTHROPIC_KEY", "DEFAULT_KEY")


    anthropic_default_model: str = "claude-sonnet-4-6"
    #anthropic_default_model: str = "claude-opus-4-6"

    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.2:latest"

    traceloop_api_key: str = "NoKey"
    traceloop_base_url: str = "http://localhost:4318"

    default_provider: Literal["anthropic", "ollama"] = "anthropic"


settings = Settings()
