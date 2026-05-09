"""Semantic section classification for scientific documents."""

from __future__ import annotations

import re


class ScientificSectionClassifier:
    """Classify section headings into a stable scientific taxonomy."""

    CANONICAL_PATTERNS = {
        "abstract": [r"^abstract$"],
        "introduction": [r"^intro(?:duction)?$"],
        "related_work": [r"^related work$", r"^literature review$"],
        "preliminaries": [r"^preliminar(?:y|ies)$", r"^background$", r"^problem setup$"],
        "methodology": [r"^method$", r"^methods$", r"^methodology$", r"^approach$"],
        "experiments": [r"^experimental setup$", r"^evaluation$", r"^benchmark$", r"^implementation details$", r"^experiments?$"],
        "results": [r"^results?$", r"^findings$", r"^ablation(?: studies)?$"],
        "discussion": [r"^discussion$"],
        "limitations": [r"^limitations?$", r"^future work$"],
        "conclusion": [r"^conclusions?$"],
        "appendix": [r"^appendix(?: [a-z0-9]+)?$", r"^supplementary material$", r"^supplemental material$"],
        "references": [r"^references$", r"^bibliography$"],
    }

    def classify(self, text: str) -> tuple[str | None, float]:
        normalized = re.sub(r"\s+", " ", text.lower()).strip(" .:")
        for label, patterns in self.CANONICAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, normalized):
                    base = 0.97 if re.fullmatch(pattern, normalized) else 0.9
                    if len(normalized.split()) <= 6:
                        base += 0.04
                    return label, min(base, 0.99)
        return None, 0.0
