"""Remove repeated phrasing and duplicated scientific statements."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.writer.revision.repetition_detector import RepetitionDetector


class RedundancyCleaner:
    """Clean repeated content at the sentence and phrase level."""

    def __init__(self) -> None:
        self.detector = RepetitionDetector()

    def clean(self, text: str) -> tuple[str, dict[str, object]]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        seen_sentences: set[str] = set()
        cleaned_paragraphs: list[str] = []
        removed_sentences = 0
        for paragraph in paragraphs:
            sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", paragraph) if segment.strip()]
            kept: list[str] = []
            for sentence in sentences:
                normalized = re.sub(r"\s+", " ", sentence.lower())
                if normalized in seen_sentences:
                    removed_sentences += 1
                    continue
                seen_sentences.add(normalized)
                kept.append(sentence)
            if kept:
                cleaned_paragraphs.append(" ".join(kept))
        cleaned = "\n\n".join(cleaned_paragraphs).strip()
        return cleaned, {
            "removed_sentences": removed_sentences,
            "repeated_ngrams": self.detector.repeated_ngrams(text),
            "discourse_diversity": self.detector.discourse_diversity(cleaned),
        }

