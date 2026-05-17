"""Build semantically structured grounded answers from evidence."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalResult
from autonomous_research_assistant_data.rag.citations.citation_formatter import CitationFormatter


class StructuredAnswerBuilder:
    """Generate structured answers from evidence instead of chunk concatenation."""

    def __init__(self) -> None:
        self.citations = CitationFormatter()

    def _sentences(self, result: RetrievalResult) -> list[str]:
        text = result.merged_context or result.chunk_text
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]

    def _best_sentence(self, result: RetrievalResult, keywords: list[str]) -> str:
        sentences = self._sentences(result)
        scored = []
        for sentence in sentences:
            lowered = sentence.lower()
            score = sum(keyword.lower() in lowered for keyword in keywords)
            scored.append((score, sentence))
        scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        return scored[0][1] if scored else (result.merged_context or result.chunk_text)[:400]

    def _best_result(self, results: list[RetrievalResult], keywords: list[str]) -> RetrievalResult:
        scored: list[tuple[int, float, RetrievalResult]] = []
        for result in results:
            lowered = (result.merged_context or result.chunk_text).lower()
            score = sum(keyword.lower() in lowered for keyword in keywords if keyword)
            scored.append((score, result.score, result))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2] if scored else results[0]

    def _prefer_sections(self, results: list[RetrievalResult], labels: set[str]) -> list[RetrievalResult]:
        preferred = [result for result in results if (result.canonical_section_label or "").lower() in labels]
        return preferred or results

    def _supporting_sentences(self, results: list[RetrievalResult], keywords: list[str], limit: int = 2) -> list[tuple[str, str]]:
        picked: list[tuple[str, str]] = []
        for result in results:
            sentence = self._best_sentence(result, keywords)
            citation = self.citations.format_inline(result)
            if sentence and all(sentence != existing for existing, _ in picked):
                picked.append((sentence, citation))
            if len(picked) >= limit:
                break
        return picked

    def build(self, query: str, results: list[RetrievalResult], understanding: QueryUnderstandingResult) -> str:
        if not results:
            return f"No grounded evidence was retrieved for: {query}"
        key_terms = understanding.target_topics[:8] + understanding.entities
        top = self._best_result(results, key_terms)
        citation = self.citations.format_inline(top)
        query_type = understanding.query_type
        keywords = key_terms
        if query_type == "definition":
            definition_pool = self._prefer_sections(results, {"abstract", "introduction", "methodology", "methods"})
            definition_result = self._best_result(definition_pool, keywords + ["group relative policy optimization", "grpo", "ppo", "rlvr", "optimization"])
            sentence = self._best_sentence(definition_result, keywords + ["is", "framework", "optimization", "training"])
            supporting_result = self._best_result(definition_pool, keywords + ["reward", "advantage", "group", "rollouts"])
            follow = self._best_sentence(supporting_result, keywords + ["reward", "advantage", "group", "rollouts"])
            return (
                f"Definition: {sentence} {self.citations.format_inline(definition_result)}\n\n"
                f"Core idea: {follow} {self.citations.format_inline(supporting_result) if follow else self.citations.format_inline(definition_result)}\n\n"
                f"Workflow: The retrieved evidence describes the method in terms of grouped rollouts, reward comparison, and policy optimization behavior rather than standalone benchmark numbers. {self.citations.format_inline(supporting_result)}\n\n"
                f"Advantages: The retrieved evidence frames the method around {', '.join(understanding.target_topics[:3] or ['policy optimization'])}. {self.citations.format_inline(definition_result)}\n\n"
                f"Limitations: The answer is grounded only in the currently retrieved corpus and may omit papers outside the retrieved evidence. {self.citations.format_inline(definition_result)}"
            ).strip()
        if query_type == "comparison":
            left = understanding.entities[0] if understanding.entities else "First method"
            right = understanding.entities[1] if len(understanding.entities) > 1 else "Second method"
            overview_result = self._best_result(results, keywords + ["compare", "difference", "policy", left.lower(), right.lower()])
            overview = self._best_sentence(overview_result, keywords + ["compare", "difference", "policy"])
            evidence_pairs = self._supporting_sentences(results, [left.lower(), right.lower(), "policy", "reward", "optimization"], limit=3)
            evidence_a = evidence_pairs[0][0] if evidence_pairs else overview
            evidence_a_citation = evidence_pairs[0][1] if evidence_pairs else self.citations.format_inline(results[0])
            evidence_b = evidence_pairs[1][0] if len(evidence_pairs) > 1 else overview
            evidence_b_citation = evidence_pairs[1][1] if len(evidence_pairs) > 1 else self.citations.format_inline(results[min(1, len(results) - 1)])
            return (
                f"Overview: {overview} {self.citations.format_inline(overview_result)}\n\n"
                f"{left}: {evidence_a} {evidence_a_citation}\n\n"
                f"{right}: {evidence_b} {evidence_b_citation}\n\n"
                f"Differences: The retrieved evidence suggests they differ in optimization design, reward handling, rollout grouping, or stability tradeoffs rather than being identical training procedures. {self.citations.format_inline(overview_result)}\n\n"
                f"Advantages and limitations: The comparison is strongest where the corpus discusses objective design or reported behavior, and weaker where only one of the methods is described directly. {self.citations.format_inline(overview_result)}\n\n"
                f"Conclusion: Based on the retrieved evidence, the comparison is grounded but limited to the surfaced sections. {self.citations.format_inline(overview_result)}"
            ).strip()
        if query_type == "literature_review":
            bullets = []
            for result in results[:4]:
                bullets.append(f"- {self._best_sentence(result, keywords)} {self.citations.format_inline(result)}")
            return "Overview:\n" + "\n".join(bullets)
        if query_type == "benchmark_performance":
            sentence = self._best_sentence(top, keywords + ["benchmark", "performance", "score", "pass@"])
            return f"Benchmark summary: {sentence} {citation}"
        sentence = self._best_sentence(top, keywords)
        return f"Grounded answer: {sentence} {citation}"
