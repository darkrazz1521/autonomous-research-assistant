"""Iterative refinement for long-form section drafts."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import AnswerQualityReport, GroundingReport, RAGAnswer, SectionDraft, SectionEvidenceRecord, WriterRevisionRecord
from autonomous_research_assistant_data.rag.answer_quality.scorer import AnswerQualityScorer
from autonomous_research_assistant_data.rag.grounding.grounding_checker import GroundingChecker
from autonomous_research_assistant_data.writer.revision.coherence_rewriter import CoherenceRewriter
from autonomous_research_assistant_data.writer.revision.grounding_repair import GroundingRepair
from autonomous_research_assistant_data.writer.revision.redundancy_cleaner import RedundancyCleaner
from autonomous_research_assistant_data.writer.revision.repetition_detector import RepetitionDetector
from autonomous_research_assistant_data.writer.revision.transition_optimizer import TransitionOptimizer
from autonomous_research_assistant_data.writer.style.style_controller import StyleController
from autonomous_research_assistant_data.writer.synthesis.paraphraser import Paraphraser


class RevisionEngine:
    """Run bounded revisions to improve coherence, grounding, and citation alignment."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.grounding = GroundingChecker(config)
        self.quality = AnswerQualityScorer(config)
        self.redundancy = RedundancyCleaner()
        self.coherence = CoherenceRewriter()
        self.transitions = TransitionOptimizer()
        self.grounding_repair = GroundingRepair()
        self.repetition = RepetitionDetector()
        self.style_controller = StyleController()
        self.paraphraser = Paraphraser()

    def _score(self, draft: SectionDraft) -> tuple[GroundingReport, AnswerQualityReport]:
        answer = RAGAnswer(query=draft.title, answer=draft.content, citations=draft.citations, evidence_chunks=draft.evidence_chunks)
        grounding = self.grounding.check(answer)
        quality = self.quality.score(answer, grounding)
        return grounding, quality

    def _anti_template(self, content: str, title: str) -> tuple[str, dict[str, object]]:
        replacements = {
            "Building on the earlier discussion of": "Extending the earlier discussion of",
            "Together, the evidence indicates that": "Across the retrieved studies, the strongest supported interpretation is that",
            "Supporting evidence appears in": "This interpretation is supported by",
            "Taken together, these observations": "Collectively, these observations",
        }
        updated = content
        rewrites = 0
        for source, target in replacements.items():
            if source in updated:
                updated = updated.replace(source, target)
                rewrites += 1
        diversified = []
        for index, paragraph in enumerate([item.strip() for item in updated.split("\n\n") if item.strip()]):
            diversified.append(self.style_controller.diversify_sentence(paragraph, title, index))
        return "\n\n".join(diversified), {"template_rewrites": rewrites, "paragraphs_reworked": len(diversified)}

    def _paraphrase_refinement(self, content: str) -> tuple[str, dict[str, object]]:
        refined = []
        compressed = 0
        for sentence in [item.strip() for item in content.split(". ") if item.strip()]:
            rewritten, meta = self.paraphraser.paraphrase_sentence(sentence)
            if rewritten:
                refined.append(rewritten.rstrip("."))
            if meta.get("compressed"):
                compressed += 1
        text = ". ".join(refined).strip()
        if text and text[-1] not in ".!?":
            text += "."
        return text, {"compressed_sentences": compressed, "sentence_count": len(refined)}

    def revise(
        self,
        draft: SectionDraft,
        evidence: SectionEvidenceRecord,
        *,
        passes: int,
        previous_title: str | None = None,
        previous_section_texts: list[str] | None = None,
        quality_controls: dict[str, bool] | None = None,
    ) -> SectionDraft:
        revised = draft.model_copy(deep=True)
        previous_section_texts = previous_section_texts or []
        quality_controls = quality_controls or {}
        before_grounding = revised.grounding_report or GroundingReport()
        before_quality = revised.answer_quality_report or AnswerQualityReport()
        current_quality = before_quality.overall_answer_quality_score
        for revision_index in range(1, min(passes, self.config.writer.max_revision_passes) + 1):
            content = revised.content
            rewrite_trace: dict[str, object] = {}
            actions: list[str] = []

            content, redundancy_metrics = self.redundancy.clean(content)
            rewrite_trace["redundancy"] = redundancy_metrics
            if int(redundancy_metrics.get("removed_sentences", 0)) > 0:
                actions.append("remove_redundant_sentences")

            content, transition_metrics = self.transitions.optimize(
                content,
                current_title=revised.title,
                previous_title=previous_title,
            )
            rewrite_trace["transitions"] = transition_metrics
            if int(transition_metrics.get("transition_rewrites", 0)) > 0:
                actions.append("rewrite_transitions")

            if quality_controls.get("anti_repetition") or quality_controls.get("discourse_refinement"):
                content, anti_template_metrics = self._anti_template(content, revised.title)
                rewrite_trace["anti_template"] = anti_template_metrics
                if int(anti_template_metrics.get("template_rewrites", 0)) > 0:
                    actions.append("discourse_refinement")

            content, grounding_metrics = self.grounding_repair.repair(content, revised.grounding_report)
            rewrite_trace["grounding"] = grounding_metrics
            if int(grounding_metrics.get("removed_unsupported_claims", 0)) > 0:
                actions.append("remove_unsupported_claims")

            if evidence.contradiction_report and evidence.contradiction_report.contradiction_score > 0 and "Uncertainty:" not in content:
                content += "\n\nUncertainty: The retrieved evidence does not collapse into a single interpretation, so this section preserves the main cross-source tensions explicitly."
                actions.append("surface_uncertainty")

            if evidence.coverage_score < self.config.writer.evidence_coverage_threshold and "remaining evidence gaps" not in content.lower():
                content += "\n\nThe remaining evidence gaps mainly concern subtopics that were only partially recovered by section-specific retrieval."
                actions.append("mark_evidence_gaps")

            content, coherence_metrics = self.coherence.rewrite(content)
            rewrite_trace["coherence"] = coherence_metrics
            if int(coherence_metrics.get("paragraph_merges", 0)) > 0:
                actions.append("restructure_paragraphs")

            if quality_controls.get("paraphrase"):
                content, paraphrase_metrics = self._paraphrase_refinement(content)
                rewrite_trace["paraphrase_refinement"] = paraphrase_metrics
                if int(paraphrase_metrics.get("compressed_sentences", 0)) > 0:
                    actions.append("paraphrase_refinement")

            revised.content = content.strip()
            grounding, quality = self._score(revised)
            revised.grounding_report = grounding
            revised.answer_quality_report = quality
            repetition_overlap = self.repetition.cross_section_overlap(revised.content, previous_section_texts)
            discourse_diversity = self.repetition.discourse_diversity(revised.content)
            repeated_ngrams = self.repetition.repeated_ngrams(revised.content)
            refinement_gain = round(quality.overall_answer_quality_score - current_quality, 6)
            coherence_score = round(
                max(
                    0.0,
                    0.45 * grounding.grounding_score
                    + 0.25 * discourse_diversity
                    + 0.15 * (1.0 - quality.redundancy_score)
                    + 0.15 * (1.0 - repetition_overlap),
                ),
                6,
            )
            revised.revision_history.append(
                WriterRevisionRecord(
                    revision_index=revision_index,
                    actions=actions or ["quality_stability_pass"],
                    refinement_gain=refinement_gain,
                    unsupported_claim_frequency=grounding.unsupported_claim_ratio,
                    redundancy_score=quality.redundancy_score,
                    coherence_score=coherence_score,
                    metadata={
                        "before_quality": current_quality,
                        "after_quality": quality.overall_answer_quality_score,
                        "before_grounding": before_grounding.grounding_score,
                        "after_grounding": grounding.grounding_score,
                        "revision_delta": {
                            "quality": refinement_gain,
                            "grounding": round(grounding.grounding_score - before_grounding.grounding_score, 6),
                            "unsupported_claims": round(before_grounding.unsupported_claim_ratio - grounding.unsupported_claim_ratio, 6),
                        },
                        "rewrite_trace": rewrite_trace,
                        "cross_section_overlap": repetition_overlap,
                        "discourse_diversity": discourse_diversity,
                        "repeated_ngrams": repeated_ngrams[:12],
                    },
                )
            )
            before_grounding = grounding
            current_quality = quality.overall_answer_quality_score
        revised.metadata.setdefault("revision_trace", [])
        revised.metadata["revision_trace"].extend(item.metadata for item in revised.revision_history[-min(passes, self.config.writer.max_revision_passes) :])
        return revised
