"""Construct coherent scientific paragraphs from claims."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalResult, WritingSectionPlan
from autonomous_research_assistant_data.writer.citations.citation_manager import CitationManager
from autonomous_research_assistant_data.writer.style.style_controller import StyleController


class ParagraphConstructor:
    """Assemble claim -> evidence -> interpretation style paragraphs."""

    def __init__(self, style_controller: StyleController | None = None) -> None:
        self.style_controller = style_controller or StyleController()

    def _comparison_sentence(self, claims: list[dict[str, object]], understanding: QueryUnderstandingResult) -> str:
        entities = understanding.entities[:2]
        if len(entities) >= 2:
            return (
                f"Across the retrieved evidence, {entities[0]} and {entities[1]} pursue related optimization goals while diverging in how they construct learning signals, regulate policy updates, and balance stability against efficiency."
            )
        return "Across the retrieved evidence, the compared methods share a common optimization objective while differing in learning signal design and empirical tradeoffs."

    def _interpretation(self, claim: dict[str, object], plan: WritingSectionPlan) -> str:
        focus_terms = [term for term in claim["focus_terms"] if len(term) > 3][:3]
        if plan.title.lower().startswith("conclusion"):
            return "Taken together, these observations motivate a cautious summary rather than a universal conclusion."
        if "comparison" in plan.title.lower():
            return (
                f"This comparison is most informative around {', '.join(focus_terms)} because those themes recur across the retrieved support."
                if focus_terms
                else "This comparison is most informative where multiple retrieved sources describe the same tradeoffs."
            )
        if "limitation" in plan.title.lower() or "open" in plan.title.lower():
            return "The retrieved evidence therefore highlights unresolved limitations and open questions rather than a fully settled consensus."
        return (
            f"Within this section, the most persistent themes concern {', '.join(focus_terms)}."
            if focus_terms
            else "The retrieved studies therefore support a coherent interpretation for this section."
        )

    def _evidence_sentence(self, evidence_parts: list[str], plan: WritingSectionPlan, citations: list[str]) -> str:
        if not evidence_parts:
            return ""
        joined = ", ".join(evidence_parts[:-1]) + (f" and {evidence_parts[-1]}" if len(evidence_parts) > 1 else evidence_parts[0])
        if "comparison" in plan.title.lower():
            base = f"The contrast is supported by evidence drawn from {joined}"
        elif "method" in plan.title.lower():
            base = f"The procedural details are documented in {joined}"
        else:
            base = f"Support for this interpretation appears in {joined}"
        if citations:
            return f"{base} {' '.join(dict.fromkeys(citations))}."
        return f"{base}."

    def build(
        self,
        *,
        topic: str,
        plan: WritingSectionPlan,
        understanding: QueryUnderstandingResult,
        claims: list[dict[str, object]],
        citation_manager: CitationManager,
        style_profile: dict[str, object],
        previous_title: str | None,
        rhetorical_plan: dict[str, object] | None = None,
    ) -> tuple[list[str], list[dict[str, object]]]:
        paragraphs: list[str] = []
        traces: list[dict[str, object]] = []
        rhetorical_plan = rhetorical_plan or {}
        paragraph_roles = list(rhetorical_plan.get("paragraph_roles", []))
        if not claims:
            transition = self.style_controller.transition(previous_title, plan.title, rhetorical_role="fallback")
            paragraphs.append(f"{transition} The currently retrieved evidence is too sparse to support a stronger scientific synthesis for this section.")
            return paragraphs, traces
        grouped = claims[: max(int(style_profile.get("paragraphs", 3)), 2)]
        transition = self.style_controller.transition(
            previous_title,
            plan.title,
            rhetorical_role="comparison" if grouped and grouped[0]["claim_type"] == "comparison" else "section_opening",
        )
        for index, claim in enumerate(grouped):
            support = claim["supporting_results"]
            claim_text = str(claim["claim_text"])
            if index == 0 and claim["claim_type"] == "comparison":
                claim_text = self._comparison_sentence(grouped, understanding)
            evidence_parts: list[str] = []
            citations: list[str] = []
            for result in support[:2]:
                assert isinstance(result, RetrievalResult)
                evidence_parts.append(self.style_controller.evidence_phrase(result.section_name, result.metadata.get("canonical_section_label")))
                citations.append(citation_manager.format_inline(result))
            interpretation = self._interpretation(claim, plan)
            paragraph_sentences = []
            if index == 0:
                paragraph_sentences.append(transition)
            paragraph_sentences.append(claim_text)
            evidence_sentence = self._evidence_sentence(evidence_parts, plan, citations)
            if evidence_sentence:
                paragraph_sentences.append(evidence_sentence)
            connector = self.style_controller.connector(plan.title)
            paragraph_sentences.append(f"{connector} {interpretation[:1].lower() + interpretation[1:] if interpretation else ''}".strip())
            paragraph = " ".join(sentence.strip() for sentence in paragraph_sentences if sentence.strip())
            paragraph = self.style_controller.diversify_sentence(paragraph, plan.title, index)
            paragraphs.append(re.sub(r"\s+", " ", paragraph).strip())
            traces.append(
                {
                    "paragraph_index": index,
                    "claim_type": claim["claim_type"],
                    "source_count": claim["source_count"],
                    "citations": citations,
                    "novelty_score": float(claim.get("novelty_score", 1.0)),
                    "paragraph_role": paragraph_roles[index] if index < len(paragraph_roles) else "analysis",
                }
            )
        return paragraphs, traces
