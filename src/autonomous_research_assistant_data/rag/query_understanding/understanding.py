"""Intent classification, entity extraction, and answer-structure prediction."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import QueryUnderstandingResult


QUERY_TYPE_RULES: list[tuple[str, tuple[str, ...], list[str]]] = [
    ("comparison", ("compare", "difference", "versus", "vs", "better than"), ["overview", "side_by_side", "differences", "advantages", "limitations", "conclusion"]),
    ("definition", ("what is", "define", "meaning of"), ["definition", "core idea", "workflow", "advantages", "limitations", "citations"]),
    ("summarization", ("summarize", "summary", "overview of"), ["overview", "key points", "takeaways", "citations"]),
    ("contradiction_analysis", ("contradiction", "conflict", "disagree", "inconsistent"), ["claim", "supporting evidence", "contradictions", "resolution", "citations"]),
    ("methodology_explanation", ("how does", "method", "approach", "algorithm"), ["method", "workflow", "training objective", "advantages", "limitations", "citations"]),
    ("benchmark_performance", ("benchmark", "performance", "result", "accuracy", "pass@"), ["benchmark", "reported numbers", "comparison", "caveats", "citations"]),
    ("citation_lookup", ("cite", "citation", "who said"), ["answer", "source", "citations"]),
    ("literature_review", ("literature review", "survey", "recent work", "papers on"), ["overview", "major approaches", "trends", "limitations", "open problems", "citations"]),
    ("timeline_history", ("timeline", "history", "evolution", "chronology"), ["timeline", "milestones", "recent shifts", "citations"]),
]

ACRONYM_MAP = {
    "grpo": ["Group Relative Policy Optimization", "policy optimization methods", "RLVR"],
    "ppo": ["Proximal Policy Optimization", "policy gradient optimization"],
    "rlvr": ["reinforcement learning with verifiable rewards"],
    "rlhf": ["reinforcement learning from human feedback"],
    "llm": ["large language model"],
}

STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
    "compare",
    "summarize",
    "define",
    "explain",
}


class QueryUnderstandingAnalyzer:
    """Infer the semantic type and structure of a research query."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _normalize(self, query: str) -> str:
        normalized = " ".join(query.strip().split())
        # Remove raw arXiv-like identifiers when they are appended as retrieval noise.
        normalized = re.sub(r"\b\d{4}\.\d{5}(?:v\d+)?\b", "", normalized, flags=re.IGNORECASE)
        return " ".join(normalized.split())

    def _query_type(self, lowered: str) -> tuple[str, list[str]]:
        for query_type, triggers, structure in QUERY_TYPE_RULES:
            if any(trigger in lowered for trigger in triggers):
                return query_type, structure
        return "definition", ["definition", "core idea", "evidence", "citations"]

    def _entities(self, query: str) -> list[str]:
        entities = re.findall(r"\b(?:[A-Z]{2,}|[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*)\b", query)
        cleaned: list[str] = []
        for entity in list(dict.fromkeys(entities)):
            normalized = re.sub(r"^(?:compare|define|explain|summarize)\s+", "", entity, flags=re.IGNORECASE).strip()
            if normalized and normalized.lower() not in STOPWORDS:
                cleaned.append(normalized)
        return cleaned

    def _target_topics(self, lowered: str) -> list[str]:
        topics = []
        for token, expansions in ACRONYM_MAP.items():
            if token in lowered:
                topics.extend([token, *expansions])
        topics.extend(
            token
            for token in re.findall(r"[a-z][a-z0-9\-]{3,}", lowered)
            if token not in STOPWORDS
        )
        return list(dict.fromkeys(topics))[:12]

    def analyze(self, query: str) -> QueryUnderstandingResult:
        normalized = self._normalize(query)
        lowered = normalized.lower()
        query_type, structure = self._query_type(lowered)
        entities = self._entities(normalized) if self.config.rag.query_understanding.entity_extraction else []
        expanded_terms: list[str] = []
        if self.config.rag.query_understanding.acronym_expansion:
            for token, expansions in ACRONYM_MAP.items():
                if token in lowered:
                    expanded_terms.extend(expansions)
        return QueryUnderstandingResult(
            normalized_query=normalized,
            expanded_terms=list(dict.fromkeys(expanded_terms)),
            entities=entities,
            query_type=query_type,
            target_topics=self._target_topics(lowered) if self.config.rag.query_understanding.topic_extraction else [],
            expected_answer_structure=structure,
            metadata={
                "query_length": len(normalized),
                "intent_classification_enabled": self.config.rag.query_understanding.intent_classification,
            },
        )
