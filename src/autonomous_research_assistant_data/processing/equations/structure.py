"""Equation structure engine v2."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import CleanParagraph, EquationBlock


class EquationStructureEngine:
    """Stricter equation segmentation and integrity scoring."""

    SYMBOL_PATTERN = re.compile(r"(=|\\sum|\\prod|\\int|\\frac|Σ|Π|∂|∇|θ|λ|β|γ|→|←|≤|≥|≈|\|\||\barg\s+min\b|\barg\s+max\b)")
    EQUATION_LINE_PATTERN = re.compile(r"^[A-Za-z0-9_(),{}\[\]\s\\|<>≤≥≈ΣΠ∂∇θλβγ=+\-*/\.]+$")
    ENUMERATION_PATTERN = re.compile(r"^\(?\d+\)?[.)]\s+[A-Z]")

    def __init__(self, repair_level: str = "balanced") -> None:
        self.repair_level = repair_level

    def _looks_like_equation(self, text: str) -> bool:
        if self.ENUMERATION_PATTERN.match(text.strip()):
            return False
        symbol_hits = len(self.SYMBOL_PATTERN.findall(text))
        if symbol_hits == 0:
            return False
        if text.count("=") >= 1 and len(text.split()) <= 40:
            return True
        if symbol_hits >= 2 and self.EQUATION_LINE_PATTERN.match(text.strip()):
            return True
        return False

    def _equation_type(self, text: str) -> str:
        lowered = text.lower()
        if "\\frac" in lowered or "/" in text:
            return "fractional"
        if "\\sum" in lowered or "Σ" in text:
            return "summation"
        if "\\prod" in lowered or "Π" in text:
            return "product"
        if "[" in text and "]" in text and ";" in text:
            return "matrix"
        return "display"

    def _integrity(self, text: str) -> float:
        open_brackets = text.count("(") + text.count("[") + text.count("{")
        close_brackets = text.count(")") + text.count("]") + text.count("}")
        bracket_score = max(0.0, 1.0 - abs(open_brackets - close_brackets) * 0.15)
        operator_score = min(1.0, len(self.SYMBOL_PATTERN.findall(text)) / 3.0)
        truncation_penalty = 0.18 if re.search(r"(=|\\sum|\\frac|≤|≥|≈)\s*$", text) else 0.0
        return round(max(0.0, min(1.0, bracket_score * 0.55 + operator_score * 0.45 - truncation_penalty)), 4)

    def repair(self, paper_id: str, arxiv_id: str, paragraphs: list[CleanParagraph]) -> tuple[list[CleanParagraph], list[EquationBlock], dict[str, int]]:
        repaired: list[CleanParagraph] = []
        equation_blocks: list[EquationBlock] = []
        idx = 0

        while idx < len(paragraphs):
            current = paragraphs[idx].model_copy(deep=True)
            current_is_equation = current.is_equation or self._looks_like_equation(current.text)
            if not current_is_equation:
                current.is_equation = False
                repaired.append(current)
                idx += 1
                continue

            merged = [current]
            next_idx = idx + 1
            while next_idx < len(paragraphs):
                candidate = paragraphs[next_idx].model_copy(deep=True)
                if not (candidate.is_equation or self._looks_like_equation(candidate.text)):
                    break
                merged.append(candidate)
                next_idx += 1

            merged_text = "\n".join(item.text for item in merged)
            integrity = self._integrity(merged_text)
            eq_type = self._equation_type(merged_text)
            block_id = f"{paper_id}-eq{len(equation_blocks):04d}"
            equation_blocks.append(
                EquationBlock(
                    equation_id=block_id,
                    paper_id=paper_id,
                    arxiv_id=arxiv_id,
                    text=merged_text,
                    normalized_text=re.sub(r"\s+", " ", merged_text).strip(),
                    page_start=merged[0].page_number,
                    page_end=merged[-1].page_end or merged[-1].page_number,
                    paragraph_ids=[item.paragraph_id for item in merged],
                    is_multiline=len(merged) > 1,
                    operator_count=len(self.SYMBOL_PATTERN.findall(merged_text)),
                    confidence=0.92 if integrity >= 0.75 else 0.74,
                    chunk_guard=True,
                    equation_integrity_score=integrity,
                    equation_type=eq_type,
                )
            )
            canonical = merged[0].model_copy(deep=True)
            canonical.text = merged_text
            canonical.is_equation = True
            canonical.equation_block_id = block_id
            canonical.equation_integrity_score = integrity
            canonical.equation_type = eq_type
            canonical.structural_role = "equation_block"
            repaired.append(canonical)
            idx = next_idx

        return repaired, equation_blocks, {
            "equation_block_count": len(equation_blocks),
            "low_integrity_equations": sum(1 for item in equation_blocks if item.equation_integrity_score < 0.7),
        }
