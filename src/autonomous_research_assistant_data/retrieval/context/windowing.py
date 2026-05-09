"""Neighboring chunk expansion for retrieval context windows."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import EmbeddingRecord, RetrievalResult
from autonomous_research_assistant_data.processing.utils import estimate_token_count


class ContextWindowBuilder:
    """Construct local semantic windows around a retrieved chunk."""

    def __init__(self, config: AppConfig, records: dict[str, EmbeddingRecord]) -> None:
        self.config = config
        self.records = records

    def _pack(self, record: EmbeddingRecord | None) -> dict[str, str]:
        if record is None:
            return {}
        return {
            "chunk_id": record.chunk_id,
            "section_name": str(record.metadata.get("section_name", "Unknown")),
            "canonical_section_label": str(record.metadata.get("canonical_section_label", "")),
            "chunk_text": record.chunk_text,
        }

    def enrich(self, result: RetrievalResult, *, enabled: bool = True, radius: int | None = None) -> RetrievalResult:
        if not enabled:
            result.primary_chunk = self._pack(self.records.get(result.chunk_id))
            result.merged_context = result.chunk_text
            return result
        cfg = self.config.retrieval.context_window
        effective_radius = radius if radius is not None else cfg.radius
        before: list[dict[str, str]] = []
        after: list[dict[str, str]] = []
        seen = {result.chunk_id}
        current = self.records.get(result.chunk_id)
        walker = current
        for _ in range(effective_radius):
            if walker is None:
                break
            prev_id = walker.metadata.get("previous_chunk_id")
            prev_record = self.records.get(str(prev_id)) if prev_id else None
            if prev_record is None or prev_record.chunk_id in seen:
                break
            seen.add(prev_record.chunk_id)
            before.insert(0, self._pack(prev_record))
            walker = prev_record
        walker = current
        for _ in range(effective_radius):
            if walker is None:
                break
            next_id = walker.metadata.get("next_chunk_id")
            next_record = self.records.get(str(next_id)) if next_id else None
            if next_record is None or next_record.chunk_id in seen:
                break
            seen.add(next_record.chunk_id)
            after.append(self._pack(next_record))
            walker = next_record
        ordered = before + [self._pack(current)] + after
        if len(ordered) > cfg.max_context_chunks:
            midpoint = len(before)
            start = max(midpoint - (cfg.max_context_chunks // 2), 0)
            ordered = ordered[start : start + cfg.max_context_chunks]
        merged_parts: list[str] = []
        running_tokens = 0
        for item in ordered:
            text = item.get("chunk_text", "")
            token_count = estimate_token_count(text)
            if merged_parts and running_tokens + token_count > cfg.max_merged_tokens:
                break
            merged_parts.append(text)
            running_tokens += token_count
        result.primary_chunk = self._pack(current)
        result.context_before = before
        result.context_after = after
        result.merged_context = "\n\n".join(part for part in merged_parts if part)
        return result
