"""Unified RAG generation service."""

from __future__ import annotations

from time import perf_counter

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.logging import get_logger
from autonomous_research_assistant_data.models.common import GenerationMetadata
from autonomous_research_assistant_data.rag.generator.backends import (
    HeuristicGenerationBackend,
    HuggingFaceGenerationBackend,
    OllamaGenerationBackend,
    OpenAICompatibleGenerationBackend,
)
from autonomous_research_assistant_data.rag.generator.models import GenerationBackendResponse, GenerationRequest


class GenerationService:
    """Route answer generation across available backends with deterministic fallback."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger("rag.generator.service")

    def _resolve_backend(self):
        provider = self.config.rag.generation.provider.lower()
        if provider == "huggingface":
            try:
                return HuggingFaceGenerationBackend(self.config.rag.generation.model_name)
            except Exception:
                pass
        if provider == "ollama" and self.config.rag.generation.api_base:
            return OllamaGenerationBackend(self.config.rag.generation.api_base)
        if provider in {"openai", "openai-compatible"} and self.config.rag.generation.api_base:
            return OpenAICompatibleGenerationBackend(self.config.rag.generation.api_base, self.config.rag.generation.api_key_env_var)
        return HeuristicGenerationBackend()

    def build_request(self, prompt: str, *, metadata: dict[str, object] | None = None, streaming: bool | None = None) -> GenerationRequest:
        gen = self.config.rag.generation
        return GenerationRequest(
            prompt=prompt,
            model_name=gen.model_name,
            provider=gen.provider,
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_tokens,
            context_window=gen.context_window,
            repetition_penalty=gen.repetition_penalty,
            stop_sequences=gen.stop_sequences,
            seed=gen.seed,
            streaming=gen.streaming if streaming is None else streaming,
            timeout_seconds=gen.timeout_seconds,
            metadata=dict(metadata or {}),
        )

    def generate(self, prompt: str, *, metadata: dict[str, object] | None = None, streaming: bool | None = None) -> tuple[str, GenerationMetadata]:
        request = self.build_request(prompt, metadata=metadata, streaming=streaming)
        backend = self._resolve_backend()
        started = perf_counter()
        response: GenerationBackendResponse
        try:
            response = backend.generate(request)
        except Exception as exc:
            self.logger.warning("Generation backend failed, using heuristic fallback", extra={"context": {"provider": request.provider, "error": str(exc)}})
            response = HeuristicGenerationBackend().generate(request)
        latency_ms = round((perf_counter() - started) * 1000, 3)
        metadata_obj = GenerationMetadata(
            provider=response.provider,
            model_name=response.model_name,
            backend=response.backend,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            streaming=request.streaming,
            seed=request.seed,
            latency_ms=latency_ms,
            prompt_chars=response.prompt_chars,
            completion_chars=response.completion_chars,
            extra=response.metadata,
        )
        return response.text, metadata_obj
