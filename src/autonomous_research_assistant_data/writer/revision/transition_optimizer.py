"""Rewrite robotic transitions into more natural scientific discourse."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.writer.style.style_controller import StyleController


class TransitionOptimizer:
    """Replace repeated transition scaffolds with section-aware alternatives."""

    def __init__(self, style_controller: StyleController | None = None) -> None:
        self.style_controller = style_controller or StyleController()

    def optimize(self, text: str, *, current_title: str, previous_title: str | None) -> tuple[str, dict[str, object]]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        replacements = 0
        for index, paragraph in enumerate(paragraphs[:2]):
            if paragraph.lower().startswith("this section examines") or paragraph.lower().startswith("building on the preceding discussion"):
                transition = self.style_controller.transition(previous_title, current_title, rhetorical_role="section_opening")
                paragraphs[index] = re.sub(
                    r"^(?:This section examines.*?|Building on the preceding discussion of .*?, this section focuses on .*?\.)\s*",
                    f"{transition} ",
                    paragraph,
                ).strip()
                replacements += 1
        return "\n\n".join(paragraphs).strip(), {"transition_rewrites": replacements}

