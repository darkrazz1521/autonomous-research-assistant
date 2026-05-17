"""Cross-section claim tracking for novelty-aware synthesis."""

from __future__ import annotations

import hashlib
import re


class ClaimGraph:
    """Track prior claims and suppress repeated scientific observations."""

    def _normalize(self, text: str) -> str:
        lowered = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
        lowered = re.sub(r"\b(?:the|this|that|these|those|with|from|into|their|study|studies|evidence)\b", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def claim_hash(self, text: str) -> str:
        return hashlib.sha256(self._normalize(text).encode("utf-8")).hexdigest()[:16]

    def similarity(self, left: str, right: str) -> float:
        left_tokens = set(self._normalize(left).split())
        right_tokens = set(self._normalize(right).split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens.intersection(right_tokens)) / max(len(left_tokens.union(right_tokens)), 1)

    def score_novelty(self, text: str, prior_claims: list[str]) -> float:
        if not prior_claims:
            return 1.0
        overlap = max(self.similarity(text, prior) for prior in prior_claims)
        return round(max(0.0, 1.0 - overlap), 6)

    def deduplicate(
        self,
        claims: list[dict[str, object]],
        *,
        prior_claims: list[str] | None = None,
        novelty_threshold: float = 0.28,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        prior_claims = prior_claims or []
        accepted: list[dict[str, object]] = []
        suppressed: list[dict[str, object]] = []
        memory = list(prior_claims)
        for claim in claims:
            text = str(claim.get("claim_text", ""))
            novelty = self.score_novelty(text, memory)
            enriched = dict(claim)
            enriched["claim_hash"] = self.claim_hash(text)
            enriched["novelty_score"] = novelty
            if novelty < novelty_threshold:
                suppressed.append({"claim_text": text, "novelty_score": novelty})
                continue
            accepted.append(enriched)
            memory.append(text)
        return accepted, {"accepted": len(accepted), "suppressed": suppressed, "prior_claim_count": len(prior_claims)}
