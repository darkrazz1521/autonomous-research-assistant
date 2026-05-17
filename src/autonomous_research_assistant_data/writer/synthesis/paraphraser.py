"""Semantic paraphrasing for grounded scientific synthesis."""

from __future__ import annotations

import re


class Paraphraser:
    """Reduce retrieval leakage while preserving scientific meaning."""

    def __init__(self) -> None:
        self._lead_rewrites = {
            "in this paper": "the study",
            "in this work": "the study",
            "we propose": "the authors propose",
            "we present": "the authors present",
            "our method": "the method",
            "results show that": "the reported results indicate that",
        }

    def _normalize(self, text: str) -> str:
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\(\s*[A-Z][A-Za-z]+(?:\s+et al\.)?,?\s*\d{4}[a-z]?\s*\)", "", text)
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip(" ,;")
        return text

    def _preserve_terminology(self, text: str, terminology: list[str]) -> str:
        preserved = text
        for term in terminology:
            if term and term.lower() in preserved.lower():
                preserved = re.sub(term, term, preserved, flags=re.IGNORECASE)
        return preserved

    def suppress_chunk_leakage(self, text: str) -> tuple[str, dict[str, object]]:
        cleaned = self._normalize(text)
        removed_equations = 0
        removed_table_fragments = 0
        if re.search(r"[=<>]{1,2}|\b(?:argmax|min|max|theta|lambda|kl)\b", cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(r"[^.]*[=<>][^.]*\.?", "", cleaned).strip()
            removed_equations += 1
        if re.search(r"\b(?:row|column|table|header)\b", cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(r"(?:\brow\b|\bcolumn\b|\bheader\b)\s+\w+", "", cleaned, flags=re.IGNORECASE).strip()
            removed_table_fragments += 1
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;")
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned, {
            "removed_equation_fragments": removed_equations,
            "removed_table_fragments": removed_table_fragments,
        }

    def paraphrase_sentence(self, sentence: str, *, terminology: list[str] | None = None) -> tuple[str, dict[str, object]]:
        rewritten = self._normalize(sentence)
        substitutions = 0
        lowered = rewritten.lower()
        for source, target in self._lead_rewrites.items():
            if source in lowered:
                rewritten = re.sub(source, target, rewritten, flags=re.IGNORECASE)
                substitutions += 1
                lowered = rewritten.lower()
        rewritten = re.sub(r"\bcan be seen\b", "is observed", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bshows?\b", "indicates", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bvery\b", "", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\s+", " ", rewritten).strip(" ,;")
        rewritten = self._preserve_terminology(rewritten, terminology or [])
        if rewritten and rewritten[-1] not in ".!?":
            rewritten += "."
        return rewritten, {"substitutions": substitutions, "compressed": len(rewritten) <= len(sentence)}

    def build_claim_sentence(self, sentences: list[str], *, terminology: list[str] | None = None) -> tuple[str, dict[str, object]]:
        if not sentences:
            return "", {"support_count": 0, "deduplicated": 0}
        normalized = []
        seen: set[str] = set()
        for sentence in sentences:
            cleaned, leakage = self.suppress_chunk_leakage(sentence)
            rewritten, meta = self.paraphrase_sentence(cleaned, terminology=terminology)
            key = rewritten.lower()
            if not rewritten or key in seen:
                continue
            seen.add(key)
            normalized.append((rewritten, {**leakage, **meta}))
        if not normalized:
            return "", {"support_count": 0, "deduplicated": len(sentences)}
        primary = normalized[0][0]
        return primary, {"support_count": len(normalized), "deduplicated": len(sentences) - len(normalized), "sentence_traces": [item[1] for item in normalized]}
