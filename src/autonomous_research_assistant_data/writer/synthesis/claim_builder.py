"""Build claim-level abstractions from clustered evidence."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalResult, WritingSectionPlan
from autonomous_research_assistant_data.writer.synthesis.paraphraser import Paraphraser


class ClaimBuilder:
    """Turn evidence clusters into grounded claims."""

    def __init__(self) -> None:
        self.paraphraser = Paraphraser()

    def _clean_anchor(self, sentence: str) -> str:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        sentence = sentence.replace("In this work, we propose", "Retrieved studies describe")
        sentence = sentence.replace("This work proposes", "Retrieved studies describe")
        sentence = sentence.replace("In contrast,", "By contrast,")
        sentence = sentence.replace("However,", "However,")
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        return sentence

    def _claim_type(self, plan: WritingSectionPlan, understanding: QueryUnderstandingResult) -> str:
        title = plan.title.lower()
        if understanding.query_type == "comparison" or "comparison" in title or "difference" in title:
            return "comparison"
        if "limitation" in title or "open" in title:
            return "limitation"
        if "introduction" in title or "overview" in title:
            return "overview"
        if "benchmark" in title or "empirical" in title or "result" in title:
            return "evidence"
        return "analysis"

    def build(
        self,
        clusters: list[dict[str, object]],
        *,
        plan: WritingSectionPlan,
        understanding: QueryUnderstandingResult,
    ) -> list[dict[str, object]]:
        claim_type = self._claim_type(plan, understanding)
        claims: list[dict[str, object]] = []
        for cluster in clusters[:8]:
            anchor, paraphrase_meta = self.paraphraser.build_claim_sentence(
                [str(sentence) for sentence in cluster["sentences"]],
                terminology=plan.required_terms,
            )
            anchor = self._clean_anchor(anchor or str(cluster["anchor"]))
            token_counter = cluster["token_counter"]
            focus_terms = [term for term, _ in token_counter.most_common(6)]
            supporting_results = []
            seen: set[str] = set()
            for result in cluster["results"]:
                assert isinstance(result, RetrievalResult)
                if result.chunk_id in seen:
                    continue
                seen.add(result.chunk_id)
                supporting_results.append(result)
            claims.append(
                {
                    "claim_text": anchor,
                    "claim_type": claim_type,
                    "focus_terms": focus_terms,
                    "supporting_results": supporting_results[:4],
                    "source_count": len(supporting_results),
                    "paraphrase_meta": paraphrase_meta,
                    "normalization_reports": list(cluster.get("normalization_reports", [])),
                }
            )
        return claims
