"""Generation data models for RAG backends."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str
    model_name: str
    provider: str
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 700
    context_window: int = 4096
    repetition_penalty: float = 1.05
    stop_sequences: list[str] = Field(default_factory=list)
    seed: int | None = None
    streaming: bool = False
    timeout_seconds: int = 60
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationBackendResponse(BaseModel):
    text: str
    backend: str
    provider: str
    model_name: str
    prompt_chars: int = 0
    completion_chars: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
