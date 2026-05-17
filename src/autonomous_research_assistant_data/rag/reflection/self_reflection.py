"""Self-reflection engine for grounded answers."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import AgenticPlan, RAGAnswer, ReflectionReport


class SelfReflectionEngine:
    """Inspect grounded answers for weak evidence and refinement needs."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def inspect(self, answer: RAGAnswer, plan: AgenticPlan) -> ReflectionReport:
        critique: list[str] = []
        actions: list[str] = []
        missing_evidence: list[str] = []
        unsupported = answer.grounding_report.unsupported_claims if answer.grounding_report else []
        if unsupported:
            critique.append("Some answer sentences are weakly grounded.")
            actions.append("retry_retrieval")
        if len(answer.evidence_chunks) < min(3, self.config.rag.synthesis.max_evidence_chunks):
            critique.append("Evidence breadth is limited.")
            missing_evidence.append("broader cross-paper evidence")
            actions.append("expand_retrieval")
        contradiction_risk = answer.grounding_report.contradiction_score if answer.grounding_report else 0.0
        if contradiction_risk > 0.1:
            critique.append("Potential disagreement across evidence should be surfaced.")
            actions.append("inspect_contradictions")
        return ReflectionReport(
            critique=critique,
            unsupported_claims=unsupported,
            missing_evidence=missing_evidence,
            contradiction_risk=round(contradiction_risk, 6),
            retrieval_sufficiency_score=round(answer.grounding_report.grounding_score if answer.grounding_report else 0.0, 6),
            refinement_actions=list(dict.fromkeys(actions)),
            metadata={"subtask_count": len(plan.subtasks)},
        )
