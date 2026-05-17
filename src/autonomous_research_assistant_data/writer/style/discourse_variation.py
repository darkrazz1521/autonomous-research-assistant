"""Discourse variation helpers for non-repetitive scientific prose."""

from __future__ import annotations

from collections import deque
import re


SECTION_POLICIES: dict[str, dict[str, list[str]]] = {
    "introduction": {
        "section_opening": [
            "The literature situates this discussion by emphasizing",
            "A useful starting point is the recurring motivation behind",
            "Recent work frames this topic through the central question of",
        ],
        "connector": [
            "More concretely,",
            "In practical terms,",
            "At a high level,",
        ],
    },
    "methods": {
        "section_opening": [
            "Methodologically, the retrieved studies converge on",
            "At the procedural level, the evidence highlights",
            "The technical core of this section concerns",
        ],
        "connector": [
            "Operationally,",
            "Mechanistically,",
            "From an implementation perspective,",
        ],
    },
    "comparison": {
        "section_opening": [
            "A clearer comparison emerges when both approaches are aligned against shared objectives.",
            "The retrieved studies support a contrastive reading of the methods rather than two isolated descriptions.",
            "The comparison becomes most informative once common assumptions are separated from design-specific choices.",
        ],
        "connector": [
            "By comparison,",
            "In contrast,",
            "On the other hand,",
            "Relative to that baseline,",
        ],
    },
    "limitations": {
        "section_opening": [
            "The available evidence also leaves several limitations unresolved.",
            "A cautious reading of the literature highlights several boundaries to these claims.",
            "Notwithstanding the reported gains, the retrieved studies expose important constraints.",
        ],
        "connector": [
            "At the same time,",
            "A key caveat is that",
            "This caution matters because",
        ],
    },
    "conclusion": {
        "section_opening": [
            "Taken together, the literature supports a bounded synthesis rather than a universal conclusion.",
            "Across the retrieved evidence, the main takeaway is a structured convergence around",
            "The strongest supported conclusion is therefore a qualified synthesis of",
        ],
        "connector": [
            "Overall,",
            "Collectively,",
            "In summary,",
        ],
    },
    "default": {
        "section_opening": [
            "The retrieved evidence supports a focused discussion of",
            "The literature provides a grounded basis for examining",
            "A recurring theme across the evidence concerns",
        ],
        "connector": [
            "Additionally,",
            "More specifically,",
            "In turn,",
        ],
    },
}


class DiscourseVariationEngine:
    """Choose varied discourse moves while tracking recent phrasing."""

    def __init__(self, memory_size: int = 8) -> None:
        self._recent_patterns: deque[str] = deque(maxlen=memory_size)

    def classify_section(self, title: str) -> str:
        lowered = title.lower()
        if "introduction" in lowered or "background" in lowered or "overview" in lowered:
            return "introduction"
        if "method" in lowered or "approach" in lowered or "procedure" in lowered:
            return "methods"
        if "comparison" in lowered or "compare" in lowered or "tradeoff" in lowered:
            return "comparison"
        if "limitation" in lowered or "caveat" in lowered or "open" in lowered:
            return "limitations"
        if "conclusion" in lowered or "takeaway" in lowered or "future" in lowered:
            return "conclusion"
        return "default"

    def _pick(self, options: list[str], *, key: str) -> str:
        for option in options:
            pattern_key = f"{key}:{option}"
            if pattern_key not in self._recent_patterns:
                self._recent_patterns.append(pattern_key)
                return option
        choice = options[len(self._recent_patterns) % max(len(options), 1)]
        self._recent_patterns.append(f"{key}:{choice}")
        return choice

    def transition(
        self,
        *,
        previous_title: str | None,
        current_title: str,
        rhetorical_role: str = "section_opening",
        section_type: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        section_type = section_type or self.classify_section(current_title)
        policy = SECTION_POLICIES.get(section_type, SECTION_POLICIES["default"])
        opening = self._pick(policy.get(rhetorical_role, policy["section_opening"]), key=f"{section_type}:{rhetorical_role}")
        current = current_title.lower()
        if previous_title:
            previous = previous_title.lower()
            if section_type == "comparison":
                sentence = f"{opening} Relative to the earlier discussion of {previous}, this section examines {current} through similarities, differences, and empirical tradeoffs."
            elif section_type == "limitations":
                sentence = f"{opening} After the preceding discussion of {previous}, the focus now shifts to the uncertainties surrounding {current}."
            else:
                sentence = f"{opening} Building from the earlier discussion of {previous}, this section develops the evidence around {current}."
        else:
            sentence = f"{opening} {current.capitalize()} is the immediate focus of this section."
        return self._clean(sentence), {"section_type": section_type, "role": rhetorical_role, "opening": opening}

    def connector(self, section_title: str) -> str:
        section_type = self.classify_section(section_title)
        policy = SECTION_POLICIES.get(section_type, SECTION_POLICIES["default"])
        return self._pick(policy["connector"], key=f"{section_type}:connector")

    def diversify_sentence_opening(self, sentence: str, section_title: str, index: int) -> str:
        stripped = sentence.strip()
        if not stripped or index == 0:
            return stripped
        starters = {"this", "the", "together", "building", "supporting", "taken"}
        first_word = re.findall(r"[A-Za-z]+", stripped[:24].lower())
        if first_word and first_word[0] in starters:
            connector = self.connector(section_title)
            stripped = re.sub(r"^[A-Za-z][A-Za-z,\s-]{0,24}", connector, stripped, count=1)
        return self._clean(stripped)

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def recent_patterns(self) -> list[str]:
        return list(self._recent_patterns)
