"""Generation backend implementations with deterministic fallback."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from importlib import import_module
from os import environ
from typing import Any

from autonomous_research_assistant_data.rag.generator.models import GenerationBackendResponse, GenerationRequest


class GenerationBackend(ABC):
    """Abstract generation backend."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationBackendResponse:
        raise NotImplementedError


class HeuristicGenerationBackend(GenerationBackend):
    """Deterministic grounded fallback when no LLM backend is available."""

    def generate(self, request: GenerationRequest) -> GenerationBackendResponse:
        context = str(request.metadata.get("compressed_context", ""))
        query = str(request.metadata.get("query", ""))
        evidence_lines = [line.strip() for line in context.splitlines() if line.strip()]
        selected = evidence_lines[: min(len(evidence_lines), 12)]
        body = " ".join(selected)
        answer = body[: request.max_tokens * 6].strip()
        if not answer:
            answer = f"No grounded answer could be synthesized for: {query}"
        return GenerationBackendResponse(
            text=answer,
            backend="heuristic",
            provider="heuristic",
            model_name=request.model_name,
            prompt_chars=len(request.prompt),
            completion_chars=len(answer),
            metadata={"deterministic": True},
        )


class HuggingFaceGenerationBackend(GenerationBackend):
    """Local transformers text-generation backend."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        transformers = import_module("transformers")
        torch = import_module("torch")
        pipeline = getattr(transformers, "pipeline")
        device = 0 if getattr(torch.cuda, "is_available")() else -1
        self._pipeline = pipeline("text-generation", model=self.model_name, device=device)

    def generate(self, request: GenerationRequest) -> GenerationBackendResponse:
        self._load()
        output = self._pipeline(  # type: ignore[operator]
            request.prompt,
            max_new_tokens=request.max_tokens,
            do_sample=request.temperature > 0,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            return_full_text=False,
        )[0]
        text = str(output.get("generated_text", "")).strip()
        return GenerationBackendResponse(
            text=text,
            backend="huggingface",
            provider="huggingface",
            model_name=request.model_name,
            prompt_chars=len(request.prompt),
            completion_chars=len(text),
        )


class OllamaGenerationBackend(GenerationBackend):
    """Ollama backend via HTTP API."""

    def __init__(self, api_base: str) -> None:
        self.api_base = api_base.rstrip("/")

    def generate(self, request: GenerationRequest) -> GenerationBackendResponse:
        payload = {
            "model": request.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repeat_penalty": request.repetition_penalty,
                "seed": request.seed,
            },
        }
        raw = _post_json(f"{self.api_base}/api/generate", payload, timeout_seconds=request.timeout_seconds)
        text = str(raw.get("response", "")).strip()
        return GenerationBackendResponse(
            text=text,
            backend="ollama",
            provider="ollama",
            model_name=request.model_name,
            prompt_chars=len(request.prompt),
            completion_chars=len(text),
            metadata={"raw": raw},
        )


class OpenAICompatibleGenerationBackend(GenerationBackend):
    """OpenAI-compatible chat completion backend."""

    def __init__(self, api_base: str, api_key_env_var: str) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key_env_var = api_key_env_var

    def generate(self, request: GenerationRequest) -> GenerationBackendResponse:
        payload = {
            "model": request.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
            "stop": request.stop_sequences or None,
            "stream": False,
        }
        headers = {}
        api_key = environ.get(self.api_key_env_var)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        raw = _post_json(f"{self.api_base}/chat/completions", payload, headers=headers, timeout_seconds=request.timeout_seconds)
        choices = raw.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        text = str(message.get("content", "")).strip()
        return GenerationBackendResponse(
            text=text,
            backend="openai-compatible",
            provider="openai-compatible",
            model_name=request.model_name,
            prompt_chars=len(request.prompt),
            completion_chars=len(text),
            metadata={"raw": raw},
        )


def _post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None, timeout_seconds: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Generation backend request failed: {exc}") from exc
