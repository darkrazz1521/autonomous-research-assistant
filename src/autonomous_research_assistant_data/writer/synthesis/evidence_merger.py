"""Merge raw retrieval evidence into cleaner semantic support clusters."""

from __future__ import annotations

import re
from collections import Counter

from autonomous_research_assistant_data.models.common import RetrievalResult
from autonomous_research_assistant_data.writer.synthesis.scientific_normalizer import ScientificNormalizer


STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "this",
    "these",
    "those",
    "their",
    "there",
    "which",
    "into",
    "such",
    "have",
    "been",
    "where",
    "when",
    "then",
    "than",
    "over",
    "under",
    "after",
    "before",
    "because",
    "while",
    "within",
    "through",
    "using",
}


class EvidenceMerger:
    """Clean and cluster evidence sentences before synthesis."""

    def __init__(self) -> None:
        self.normalizer = ScientificNormalizer()

    def _sentences(self, text: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if segment.strip()]

    def _looks_noisy(self, sentence: str) -> bool:
        stripped = sentence.strip()
        if len(stripped) < 40:
            return True
        if stripped.count("=") >= 2 or stripped.count("{") >= 1 or stripped.count("}") >= 1:
            return True
        if re.search(r"\b(?:figure|table)\s+\d+\b", stripped, flags=re.IGNORECASE):
            return True
        if re.search(r"\(\d+\)$", stripped):
            return True
        if re.search(r"(?:^|\s)[A-Z]\s*[×x]\s*[A-Z](?:\s|$)", stripped):
            return True
        if len(re.findall(r"\d", stripped)) > max(len(stripped) // 5, 12):
            return True
        if stripped.count("  ") > 2:
            return True
        return False

    def _normalize(self, sentence: str) -> str:
        sentence = re.sub(r"\s+", " ", sentence.replace("\n", " ")).strip()
        sentence = re.sub(r"\[[^\]]+\]", "", sentence).strip()
        sentence = re.sub(r"\(\s*[A-Z][A-Za-z]+ et al\.,?\s*\d{4}[a-z]?\s*\)", "", sentence).strip()
        sentence = re.sub(r"\(\s*[A-Z][A-Za-z]+(?:\s+et al\.)?,?\s*\d{4}[a-z]?\s*\)", "", sentence).strip()
        sentence = re.sub(r"\s+", " ", sentence).strip(" ,;")
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    def _tokens(self, sentence: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z][a-z0-9\-]{2,}", sentence.lower())
            if token not in STOPWORDS
        }

    def _similarity(self, left: str, right: str) -> float:
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens.intersection(right_tokens)) / max(len(left_tokens.union(right_tokens)), 1)

    def clean_sentences(self, results: list[RetrievalResult], *, limit_per_chunk: int = 4) -> list[dict[str, object]]:
        cleaned: list[dict[str, object]] = []
        for result in results:
            source = result.merged_context or result.chunk_text
            seen_local: set[str] = set()
            count = 0
            for sentence in self._sentences(source):
                normalized_fragment, fragment_report = self.normalizer.normalize_fragment(sentence)
                if not normalized_fragment:
                    continue
                if self._looks_noisy(normalized_fragment):
                    continue
                normalized = self._normalize(normalized_fragment)
                lowered = normalized.lower()
                if lowered in seen_local:
                    continue
                seen_local.add(lowered)
                cleaned.append(
                    {
                        "sentence": normalized,
                        "result": result,
                        "tokens": self._tokens(normalized),
                        "score": result.score,
                        "normalization_report": fragment_report,
                    }
                )
                count += 1
                if count >= limit_per_chunk:
                    break
        return cleaned

    def cluster(self, results: list[RetrievalResult]) -> list[dict[str, object]]:
        cleaned = self.clean_sentences(results)
        clusters: list[dict[str, object]] = []
        for item in cleaned:
            sentence = str(item["sentence"])
            assigned = False
            for cluster in clusters:
                anchor = str(cluster["anchor"])
                if self._similarity(sentence, anchor) >= 0.32:
                    cluster["sentences"].append(sentence)
                    cluster["results"].append(item["result"])
                    cluster["score"] = max(float(cluster["score"]), float(item["score"]))
                    cluster["token_counter"].update(item["tokens"])
                    cluster["normalization_reports"].append(item["normalization_report"])
                    assigned = True
                    break
            if assigned:
                continue
            clusters.append(
                {
                    "anchor": sentence,
                    "sentences": [sentence],
                    "results": [item["result"]],
                    "score": float(item["score"]),
                    "token_counter": Counter(item["tokens"]),
                    "normalization_reports": [item["normalization_report"]],
                }
            )
        clusters.sort(key=lambda cluster: (float(cluster["score"]), len(cluster["sentences"])), reverse=True)
        return clusters

    def deduplicate_claims(self, clusters: list[dict[str, object]]) -> list[dict[str, object]]:
        filtered: list[dict[str, object]] = []
        for cluster in clusters:
            anchor = str(cluster["anchor"])
            if any(self._similarity(anchor, str(existing["anchor"])) >= 0.55 for existing in filtered):
                continue
            filtered.append(cluster)
        return filtered
