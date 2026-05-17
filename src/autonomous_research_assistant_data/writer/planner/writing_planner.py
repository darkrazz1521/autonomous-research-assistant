"""Section sequencing and retrieval planning for the writer agent."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import OutlinePlan, OutlineSectionRecord, QueryUnderstandingResult, WritingPlan, WritingSectionPlan
from autonomous_research_assistant_data.writer.planner.rhetorical_planner import RhetoricalPlanner


class WritingPlanner:
    """Turn an outline into an execution plan with dependencies and section queries."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.rhetorical_planner = RhetoricalPlanner()

    def _estimate_sufficiency(self, section: OutlineSectionRecord) -> float:
        title = section.title.lower()
        if "introduction" in title or "overview" in title:
            return 0.55
        if "comparison" in title or "analysis" in title:
            return 0.72
        if "limitations" in title or "open" in title:
            return 0.62
        return 0.66

    def _section_query(self, topic: str, section: OutlineSectionRecord, understanding: QueryUnderstandingResult) -> str:
        evidence_terms = " ".join(section.required_evidence[:3])
        topic_terms = " ".join(understanding.target_topics[:4])
        return " ".join(part for part in [topic, section.title, evidence_terms, topic_terms] if part).strip()

    def _flatten(self, sections: list[OutlineSectionRecord]) -> list[OutlineSectionRecord]:
        ordered: list[OutlineSectionRecord] = []
        for section in sections:
            ordered.append(section)
            ordered.extend(self._flatten(section.subsections))
        return ordered

    def build(self, topic: str, report_type: str, outline: OutlinePlan, understanding: QueryUnderstandingResult) -> WritingPlan:
        flattened = self._flatten(outline.sections)
        sequence: list[WritingSectionPlan] = []
        prior_ids: list[str] = []
        rhetorical_traces: dict[str, object] = {}
        for section in flattened[: self.config.writer.max_sections]:
            rhetorical_plan = self.rhetorical_planner.plan(
                WritingSectionPlan(section_id=section.section_id, title=section.title, objective=section.objective),
                understanding,
            )
            sequence.append(
                WritingSectionPlan(
                    section_id=section.section_id,
                    title=section.title,
                    objective=section.objective,
                    section_query=self._section_query(topic, section, understanding),
                    retrieval_strategy=section.retrieval_strategy,
                    dependencies=prior_ids[-1:] if prior_ids else [],
                    required_terms=list(dict.fromkeys([*section.required_evidence, *understanding.entities[:2], *understanding.target_topics[:4]])),
                    evidence_sufficiency_estimate=self._estimate_sufficiency(section),
                    subsection_ids=[child.section_id for child in section.subsections],
                    metadata={"query_type": understanding.query_type, "rhetorical_plan": rhetorical_plan},
                )
            )
            rhetorical_traces[section.section_id] = rhetorical_plan
            prior_ids.append(section.section_id)
        return WritingPlan(
            topic=topic,
            report_type=report_type,
            title=outline.title,
            section_sequence=sequence,
            iterative_drafting_enabled=True,
            metadata={
                "section_count": len(sequence),
                "outline_missing_evidence_areas": outline.missing_evidence_areas,
                "rhetorical_traces": rhetorical_traces,
            },
        )
