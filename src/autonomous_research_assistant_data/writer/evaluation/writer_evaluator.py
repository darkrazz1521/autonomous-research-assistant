"""Evaluation framework for long-form writer outputs."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import ResearchReport, WriterEvaluationReport


class WriterEvaluator:
    """Score report quality at the section and document level."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _avg(self, values: list[float]) -> float:
        return round(sum(values) / max(len(values), 1), 6)

    def _readability(self, text: str) -> float:
        sentences = len(re.split(r"(?<=[.!?])\s+", text.strip())) if text.strip() else 0
        sentences = max(sentences, 1)
        words = re.findall(r"\w+", text)
        avg_sentence = len(words) / max(sentences, 1)
        if avg_sentence <= 10:
            return 0.55
        if avg_sentence <= 24:
            return 0.9
        if avg_sentence <= 34:
            return 0.75
        return 0.6

    def _discourse_diversity(self, text: str) -> float:
        tokens = re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower())
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / max(len(tokens), 1), 6)

    def evaluate(self, report: ResearchReport) -> WriterEvaluationReport:
        section_coherence = []
        grounding = []
        citation_correctness = []
        redundancy = []
        evidence_completeness = []
        synthesis_quality = []
        transition_quality = []
        readability = []
        discourse_quality = []
        revision_effectiveness = []
        paragraph_flow = []
        citation_grounding_strength = []
        rhetorical_variation = []
        claim_novelty = []
        section_individuality = []
        equation_leakage = []
        template_repetition = []
        narrative_smoothness = []
        term_sets = []
        all_section_texts = [section.content or "" for section in report.sections]
        for section in report.sections:
            content = section.content or ""
            paragraphs = [paragraph.strip() for paragraph in content.split("\n\n") if paragraph.strip()]
            if section.answer_quality_report:
                coherence_value = max(0.0, 1.0 - section.answer_quality_report.redundancy_score)
                section_coherence.append(coherence_value)
                redundancy.append(section.answer_quality_report.redundancy_score)
                synthesis_quality.append(section.answer_quality_report.synthesis_quality)
            if section.grounding_report:
                grounding.append(section.grounding_report.grounding_score)
                evidence_completeness.append(1.0 - section.grounding_report.unsupported_claim_ratio)
            citation_correctness.append(min(len(section.citations) / max(len(section.evidence_chunks), 1), 1.0))
            transition_quality.append(0.9 if len(paragraphs) >= 2 else 0.72)
            readability.append(self._readability(content))
            paragraph_flow.append(0.9 if len(paragraphs) >= 2 else 0.7)
            discourse_quality.append(self._discourse_diversity(content))
            term_sets.append(set(term.lower() for term in section.terminology))
            if section.revision_history:
                revision_effectiveness.append(max(item.refinement_gain for item in section.revision_history))
            else:
                revision_effectiveness.append(0.0)
            if section.citations:
                citation_grounding_strength.append(
                    sum(float(citation.metadata.get("grounding_confidence", 0.0)) for citation in section.citations) / max(len(section.citations), 1)
                )
            else:
                citation_grounding_strength.append(0.0)
            paragraph_roles = list(section.metadata.get("rhetorical_plan", {}).get("paragraph_roles", []))
            rhetorical_variation.append(min(len(set(paragraph_roles)) / max(len(paragraph_roles), 1), 1.0) if paragraph_roles else 0.65)
            novelty_scores = [float(item.get("novelty_score", 1.0)) for item in section.metadata.get("synthesis_trace", {}).get("paragraph_trace", [])]
            claim_novelty.append(self._avg(novelty_scores) if novelty_scores else 0.8)
            overlap_with_others = []
            current_tokens = set(re.findall(r"[a-z][a-z0-9\-]{3,}", content.lower()))
            for other_text in all_section_texts:
                if other_text == content:
                    continue
                other_tokens = set(re.findall(r"[a-z][a-z0-9\-]{3,}", other_text.lower()))
                overlap_with_others.append(len(current_tokens.intersection(other_tokens)) / max(len(current_tokens.union(other_tokens)), 1) if current_tokens and other_tokens else 0.0)
            section_individuality.append(round(1.0 - max(overlap_with_others, default=0.0), 6))
            equation_leakage.append(1.0 if re.search(r"(?:=|\\|theta|lambda|\btable\b\s+\d+)", content, flags=re.IGNORECASE) else 0.0)
            template_repetition.append(1.0 if any(pattern in content for pattern in ("Building on the earlier discussion", "Supporting evidence appears in")) else 0.0)
            narrative_smoothness.append(round(min((self._readability(content) * 0.5) + (self._discourse_diversity(content) * 0.5), 1.0), 6))
        shared_terms = set.intersection(*term_sets) if term_sets else set()
        term_union = set().union(*term_sets) if term_sets else set()
        structural = min(len(report.sections) / max(self.config.writer.max_sections, 1), 1.0)
        evaluation = WriterEvaluationReport(
            evaluation_id=f"writer-{utc_now().strftime('%Y%m%d%H%M%S')}",
            session_id=report.session_id,
            topic=report.topic,
            report_type=report.report_type,
            section_coherence=self._avg(section_coherence),
            grounding_quality=self._avg(grounding),
            citation_correctness=self._avg(citation_correctness),
            redundancy_score=self._avg(redundancy),
            terminology_consistency=round(min(len(shared_terms) / max(len(term_union), 1), 1.0), 6) if term_sets else 0.0,
            evidence_completeness=self._avg(evidence_completeness),
            structural_completeness=round(structural, 6),
            transition_quality=self._avg(transition_quality),
            synthesis_quality=self._avg(synthesis_quality),
            metadata={
                "section_count": len(report.sections),
                "semantic_coherence": self._avg(section_coherence),
                "discourse_quality": self._avg(discourse_quality),
                "redundancy_severity": self._avg(redundancy),
                "synthesis_originality": round(max(self._avg(discourse_quality) - self._avg(redundancy), 0.0), 6),
                "paragraph_flow_quality": self._avg(paragraph_flow),
                "citation_grounding_strength": self._avg(citation_grounding_strength),
                "revision_effectiveness": self._avg(revision_effectiveness),
                "scientific_readability": self._avg(readability),
                "discourse_diversity": self._avg(discourse_quality),
                "rhetorical_variation": self._avg(rhetorical_variation),
                "paraphrase_originality": round(max(self._avg(claim_novelty) - self._avg(template_repetition), 0.0), 6),
                "claim_novelty": self._avg(claim_novelty),
                "section_individuality": self._avg(section_individuality),
                "equation_leakage_severity": self._avg(equation_leakage),
                "template_repetition_severity": self._avg(template_repetition),
                "narrative_smoothness": self._avg(narrative_smoothness),
            },
        )
        return evaluation
