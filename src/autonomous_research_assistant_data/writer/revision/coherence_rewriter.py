"""Restructure paragraphs to improve scientific flow."""

from __future__ import annotations


class CoherenceRewriter:
    """Perform light paragraph restructuring for better coherence."""

    def rewrite(self, text: str) -> tuple[str, dict[str, object]]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        rewritten: list[str] = []
        merges = 0
        for paragraph in paragraphs:
            if rewritten and len(paragraph.split()) < 18:
                rewritten[-1] = f"{rewritten[-1]} {paragraph}".strip()
                merges += 1
                continue
            rewritten.append(paragraph)
        return "\n\n".join(rewritten).strip(), {"paragraph_merges": merges, "paragraph_count": len(rewritten)}

