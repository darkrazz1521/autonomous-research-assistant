"""Discourse-level controls for academic writing variation."""

from __future__ import annotations


class DiscourseController:
    """Provide varied academic discourse moves for sections and comparisons."""

    OPENINGS = [
        "The retrieved literature frames this section around a recurring scientific theme.",
        "A consistent pattern in the retrieved evidence motivates the discussion in this section.",
        "The retrieved studies provide a grounded basis for the argument developed here.",
    ]

    COMPARISON_OPENINGS = [
        "The comparison is clearest when the methods are aligned around their optimization objectives and evidence patterns.",
        "The retrieved evidence supports a structured comparison of methodological design and empirical tradeoffs.",
        "A useful comparison emerges once shared assumptions are separated from method-specific design choices.",
    ]

    def opening(self, *, comparison: bool, index: int = 0) -> str:
        pool = self.COMPARISON_OPENINGS if comparison else self.OPENINGS
        return pool[index % len(pool)]

