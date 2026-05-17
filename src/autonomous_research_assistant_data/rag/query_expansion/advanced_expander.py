"""Domain-aware multi-query expansion and HyDE-style support."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import QueryUnderstandingResult


DOMAIN_MAP = {
    "grpo": [
        "Group Relative Policy Optimization",
        "RLVR",
        "reinforcement learning with verifiable rewards",
        "PPO comparison",
        "policy optimization methods",
    ],
    "ppo": [
        "Proximal Policy Optimization",
        "policy gradient clipping",
        "actor critic baseline",
    ],
    "rlvr": [
        "reinforcement learning with verifiable rewards",
        "verifiable reward optimization",
        "reasoning RL",
    ],
    "agentic rl": [
        "long-horizon agentic reinforcement learning",
        "interactive tool-using agents",
    ],
}


class AdvancedQueryExpander:
    """Build semantically richer queries for scientific retrieval."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _semantic_terms(self, understanding: QueryUnderstandingResult) -> list[str]:
        terms = list(understanding.expanded_terms)
        lowered = understanding.normalized_query.lower()
        for key, mapped in DOMAIN_MAP.items():
            if key in lowered:
                terms.extend(mapped)
        if understanding.query_type == "comparison":
            terms.extend(["differences", "advantages", "limitations", "results"])
        if understanding.query_type == "definition":
            terms.extend(["definition", "overview", "core idea"])
        if "grpo" in lowered:
            terms.extend(
                [
                    "group-based advantage normalization",
                    "relative policy optimization",
                    "reasoning reward optimization",
                ]
            )
        if "ppo" in lowered:
            terms.extend(["surrogate objective", "clipped objective", "policy update stability"])
        return list(dict.fromkeys(term for term in terms if term))

    def _related_concepts(self, understanding: QueryUnderstandingResult) -> list[str]:
        normalized = understanding.normalized_query.lower()
        related: list[str] = []
        if any(token in normalized for token in ("grpo", "ppo", "rlvr", "rlhf")):
            related.extend(["policy optimization", "reinforcement learning", "training objective"])
        if understanding.query_type == "comparison":
            related.extend(["tradeoffs", "sample efficiency", "optimization stability"])
        if understanding.query_type == "benchmark_performance":
            related.extend(["evaluation metrics", "reported results", "ablation"])
        return list(dict.fromkeys(related))

    def expand(self, understanding: QueryUnderstandingResult, *, enable_hyde: bool = False) -> dict[str, object]:
        semantic_terms = self._semantic_terms(understanding)
        semantic_terms.extend(self._related_concepts(understanding))
        semantic_terms = list(dict.fromkeys(semantic_terms))
        rewritten_query = understanding.normalized_query
        if semantic_terms:
            rewritten_query = f"{understanding.normalized_query} {' '.join(semantic_terms[:8])}"
        multi_queries = [understanding.normalized_query]
        for term in semantic_terms[:4]:
            multi_queries.append(f"{understanding.normalized_query} {term}")
        if understanding.entities:
            multi_queries.extend(
                f"{' '.join(understanding.entities)} {term}"
                for term in semantic_terms[:2]
            )
        hyde = ""
        if enable_hyde:
            hyde = (
                f"Hypothetical answer: {understanding.normalized_query} is related to "
                f"{', '.join(semantic_terms[:4] or understanding.target_topics[:4])}."
            )
            multi_queries.append(hyde)
        multi_queries = [
            re.sub(r"\s+", " ", item).strip()
            for item in list(dict.fromkeys(multi_queries))
            if item and item.strip()
        ]
        return {
            "original_query": understanding.normalized_query,
            "expanded_terms": semantic_terms,
            "rewritten_query": rewritten_query,
            "multi_queries": multi_queries,
            "hyde_text": hyde,
        }
