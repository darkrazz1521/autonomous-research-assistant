"""Equation preservation and normalization for scientific PDFs."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import CleanParagraph, EquationBlock


class EquationRepairEngine:
    """Repair multiline equations and normalize scientific operator text."""

    EQUATION_PATTERN = re.compile(
        r"(=|\\sum|\\prod|\\int|\\frac|\\alpha|\\beta|\\gamma|\\lambda|\\theta|\\partial|\\nabla|Σ|Π|∂|∇|θ|λ|β|γ|→|←|≤|≥|≈|\bO\([^)]+\))"
    )
    CONTINUATION_PATTERN = re.compile(r"(=|\\|/|\+|-|\*|→|←|≤|≥|≈|,)\s*$")
    LEADING_CONTINUATION_PATTERN = re.compile(r"^\s*(=|\\|/|\+|-|\*|→|←|≤|≥|≈)")
    UNICODE_FIXES = {
        "â‰¤": "≤",
        "â‰¥": "≥",
        "â‰ˆ": "≈",
        "âˆ‘": "Σ",
        "âˆ": "Π",
        "âˆ‚": "∂",
        "âˆ‡": "∇",
        "âˆ’": "−",
        "Ã—": "×",
        "ﬁ": "fi",
        "ﬂ": "fl",
    }

    def normalize_math_text(self, text: str) -> str:
        normalized = text
        for broken, fixed in self.UNICODE_FIXES.items():
            normalized = normalized.replace(broken, fixed)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\s*([=+\-*/<>≤≥≈])\s*", r" \1 ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def looks_like_equation(self, text: str) -> bool:
        return bool(self.EQUATION_PATTERN.search(text)) and (
            len(text.split()) <= 40
            or text.count("=") >= 1
            or text.count("(") >= 2
            or text.count("\\") >= 1
        )

    def _should_merge_equations(self, left: CleanParagraph, right: CleanParagraph) -> bool:
        if not (left.is_equation or self.looks_like_equation(left.text)):
            return False
        if not (right.is_equation or self.looks_like_equation(right.text)):
            return False
        return (
            left.page_end == right.page_number
            or left.page_number == right.page_number
            or self.CONTINUATION_PATTERN.search(left.text) is not None
            or self.LEADING_CONTINUATION_PATTERN.search(right.text) is not None
        )

    def repair(self, paper_id: str, arxiv_id: str, paragraphs: list[CleanParagraph]) -> tuple[list[CleanParagraph], list[EquationBlock], dict[str, int]]:
        repaired: list[CleanParagraph] = []
        equation_blocks: list[EquationBlock] = []
        idx = 0
        merged_blocks = 0
        normalized_count = 0

        while idx < len(paragraphs):
            paragraph = paragraphs[idx].model_copy(deep=True)
            original_text = paragraph.text
            paragraph.text = self.normalize_math_text(paragraph.text)
            if paragraph.text != original_text:
                normalized_count += 1
            paragraph.is_equation = paragraph.is_equation or self.looks_like_equation(paragraph.text)

            if paragraph.is_equation:
                merged = [paragraph]
                next_idx = idx + 1
                while next_idx < len(paragraphs):
                    candidate = paragraphs[next_idx].model_copy(deep=True)
                    candidate_original = candidate.text
                    candidate.text = self.normalize_math_text(candidate.text)
                    if candidate.text != candidate_original:
                        normalized_count += 1
                    candidate.is_equation = candidate.is_equation or self.looks_like_equation(candidate.text)
                    if not self._should_merge_equations(merged[-1], candidate):
                        break
                    merged.append(candidate)
                    next_idx += 1

                if len(merged) > 1:
                    merged_blocks += 1
                merged_text = "\n".join(item.text for item in merged)
                normalized_text = self.normalize_math_text(merged_text.replace("\n", " "))
                equation_block_id = f"{paper_id}-eq{len(equation_blocks):04d}"
                equation_block = EquationBlock(
                    equation_id=equation_block_id,
                    paper_id=paper_id,
                    arxiv_id=arxiv_id,
                    text=merged_text,
                    normalized_text=normalized_text,
                    page_start=merged[0].page_number,
                    page_end=merged[-1].page_end or merged[-1].page_number,
                    paragraph_ids=[item.paragraph_id for item in merged],
                    is_multiline=len(merged) > 1,
                    operator_count=len(re.findall(r"[=+\-*/<>≤≥≈]", normalized_text)),
                    confidence=0.92 if len(merged) > 1 else 0.8,
                )
                first = merged[0].model_copy(deep=True)
                first.text = merged_text
                first.page_end = merged[-1].page_end or merged[-1].page_number
                first.is_equation = True
                first.equation_block_id = equation_block_id
                first.structural_role = "equation_block"
                first.repair_confidence = max(first.repair_confidence, equation_block.confidence)
                first.metadata["merged_equation_paragraph_ids"] = equation_block.paragraph_ids
                repaired.append(first)
                equation_blocks.append(equation_block)
                idx = next_idx
                continue

            if self.looks_like_equation(paragraph.text):
                paragraph.is_equation = True
                paragraph.structural_role = "equation_inline"
            repaired.append(paragraph)
            idx += 1

        stats = {
            "equation_block_count": len(equation_blocks),
            "merged_equation_blocks": merged_blocks,
            "normalized_equation_paragraphs": normalized_count,
        }
        return repaired, equation_blocks, stats
