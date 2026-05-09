"""Local heuristic query expansion for scientific retrieval."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig


EXPANSION_MAP = {
    "grpo": ["Group Relative Policy Optimization", "RLHF GRPO", "reinforcement learning optimization"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "llm": ["large language model", "large language models"],
    "rlhf": ["reinforcement learning from human feedback"],
}

SYNONYMS = {
    "methodology": ["method", "approach"],
    "results": ["findings", "outcomes"],
    "benchmark": ["evaluation", "leaderboard"],
}


class QueryExpander:
    """Build expanded query forms without external services."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def expand(self, query: str, *, enabled: bool = True) -> dict[str, object]:
        terms: list[str] = []
        if not enabled:
            return {"original_query": query, "expanded_terms": terms, "rewritten_query": query}
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", query)
        lowered = [token.lower() for token in tokens]
        for token in lowered:
            terms.extend(EXPANSION_MAP.get(token, []))
            if self.config.retrieval.query_expansion.normalize_plurals and token.endswith("s") and token[:-1] in EXPANSION_MAP:
                terms.extend(EXPANSION_MAP[token[:-1]])
            for canonical, synonym_terms in SYNONYMS.items():
                if token == canonical or token in synonym_terms:
                    terms.extend([canonical, *synonym_terms])
        expanded_terms = sorted(dict.fromkeys(term for term in terms if term and term.lower() not in query.lower()))
        rewritten = query if not expanded_terms else f"{query} {' '.join(expanded_terms)}"
        return {"original_query": query, "expanded_terms": expanded_terms, "rewritten_query": rewritten}
