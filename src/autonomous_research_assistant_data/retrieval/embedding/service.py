"""Embedding service abstraction with runtime fallback."""

from __future__ import annotations

import math
import re
from importlib import import_module

import numpy as np


class EmbeddingService:
    """Generate scientific text embeddings with optional sentence-transformers acceleration."""

    def __init__(
        self,
        model_name: str,
        *,
        normalize: bool = True,
        max_length: int = 512,
        device: str = "auto",
        fallback_dim: int = 768,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.max_length = max_length
        self.device = device
        self.fallback_dim = fallback_dim
        self._model = None
        self._backend = "deterministic-hash"
        self._vector_dim = fallback_dim
        self._load_backend()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    def _load_backend(self) -> None:
        try:
            sentence_transformers = import_module("sentence_transformers")
            SentenceTransformer = getattr(sentence_transformers, "SentenceTransformer")
            device = self._resolve_device()
            self._model = SentenceTransformer(self.model_name, device=device)
            self._backend = "sentence-transformers"
            self._vector_dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self._model = None
            self._backend = "deterministic-hash"
            self._vector_dim = self.fallback_dim

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            torch = import_module("torch")
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _prepare_text(self, text: str, *, query: bool) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if self.model_name.startswith("intfloat/e5"):
            prefix = "query: " if query else "passage: "
            return prefix + cleaned
        return cleaned

    def _hash_embed(self, texts: list[str], *, query: bool) -> np.ndarray:
        vectors = np.zeros((len(texts), self.fallback_dim), dtype=np.float32)
        for row, text in enumerate(texts):
            prepared = self._prepare_text(text, query=query).lower()
            tokens = re.findall(r"[a-z0-9_\-\.\(\)\[\]\\/%=:+*]+", prepared)
            if not tokens:
                continue
            for token in tokens:
                digest = abs(hash(token))
                index = digest % self.fallback_dim
                sign = -1.0 if (digest // self.fallback_dim) % 2 else 1.0
                vectors[row, index] += sign * (1.0 + min(len(token), 12) / 12.0)
            vectors[row] /= max(math.sqrt(len(tokens)), 1.0)
        return self._normalize(vectors)

    def encode(self, texts: list[str], *, query: bool = False, batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        if self._model is None:
            return self._hash_embed(texts, query=query)
        prepared = [self._prepare_text(text, query=query) for text in texts]
        vectors = self._model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors, dtype=np.float32)
        return vectors.astype(np.float32)

