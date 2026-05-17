"""Retrieval-aware prompt construction."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalResult
from autonomous_research_assistant_data.rag.prompts.templates import TEMPLATES


class RetrievalAwarePromptBuilder:
    """Compress retrieval results into citation-traceable prompts."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        seen: set[str] = set()
        unique: list[RetrievalResult] = []
        for result in sorted(results, key=lambda item: item.score, reverse=True):
            key = str(result.metadata.get("semantic_hash") or result.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(result)
        return unique

    def _format_context(self, results: list[RetrievalResult], *, max_context_chunks: int | None = None) -> tuple[str, list[RetrievalResult]]:
        max_chunks = max_context_chunks or self.config.rag.prompts.max_context_chunks
        kept = self._deduplicate(results)[:max_chunks]
        blocks: list[str] = []
        total_chars = 0
        section_symbol = "\u00A7"
        for item in kept:
            citation = f"[{item.paper_id} {section_symbol}{item.section_name}]"
            content = item.merged_context or item.chunk_text
            block = f"{citation}\n{content.strip()}"
            if blocks and total_chars + len(block) > self.config.rag.prompts.max_prompt_chars:
                break
            blocks.append(block)
            total_chars += len(block)
        return "\n\n".join(blocks), kept[: len(blocks)]

    def _resolve_prompt_type(self, prompt_type: str, understanding: QueryUnderstandingResult | None) -> str:
        if prompt_type != "qa":
            return prompt_type
        if not understanding:
            return prompt_type
        mapping = {
            "comparison": "comparison",
            "summarization": "summarization",
            "literature_review": "literature_review",
            "contradiction_analysis": "contradiction",
        }
        return mapping.get(understanding.query_type, prompt_type)

    def build(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        prompt_type: str = "qa",
        understanding: QueryUnderstandingResult | None = None,
        max_context_chunks: int | None = None,
    ) -> dict[str, object]:
        context, kept = self._format_context(results, max_context_chunks=max_context_chunks)
        resolved_prompt_type = self._resolve_prompt_type(prompt_type, understanding)
        template = TEMPLATES.get(resolved_prompt_type, TEMPLATES["qa"])
        prompt = template.format(query=query, context=context)
        return {
            "prompt": prompt,
            "compressed_context": context,
            "kept_results": kept,
            "prompt_type": resolved_prompt_type,
        }
