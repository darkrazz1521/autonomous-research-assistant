"""Outline generation for long-form scientific reports."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import OutlinePlan, OutlineSectionRecord, QueryUnderstandingResult


REPORT_TEMPLATES: dict[str, list[dict[str, object]]] = {
    "literature-review": [
        {"title": "Introduction", "objective": "Frame the research topic and scope.", "required_evidence": ["definitions", "motivation"]},
        {"title": "Core Concepts", "objective": "Define the main concepts, terminology, and formulations.", "required_evidence": ["definitions", "methods"]},
        {"title": "Major Approaches", "objective": "Compare the main approaches in the retrieved literature.", "required_evidence": ["approaches", "comparisons"]},
        {"title": "Empirical Evidence", "objective": "Summarize reported results, strengths, and caveats.", "required_evidence": ["results", "benchmarks"]},
        {"title": "Limitations and Open Questions", "objective": "Surface unresolved issues and evidence gaps.", "required_evidence": ["limitations", "open problems"]},
    ],
    "technical-summary": [
        {"title": "Overview", "objective": "Summarize the technical idea and motivation.", "required_evidence": ["overview", "motivation"]},
        {"title": "Mechanism", "objective": "Explain how the method or system works.", "required_evidence": ["methodology", "workflow"]},
        {"title": "Reported Evidence", "objective": "Summarize the strongest supporting evidence.", "required_evidence": ["results", "supporting evidence"]},
    ],
    "comparison": [
        {"title": "Introduction", "objective": "Frame the comparison criteria and scope.", "required_evidence": ["definitions", "comparison criteria"]},
        {"title": "Method A", "objective": "Summarize the first method.", "required_evidence": ["method A", "objective"]},
        {"title": "Method B", "objective": "Summarize the second method.", "required_evidence": ["method B", "objective"]},
        {"title": "Comparative Analysis", "objective": "Compare differences, tradeoffs, and evidence.", "required_evidence": ["tradeoffs", "results", "limitations"]},
        {"title": "Conclusion", "objective": "Summarize the grounded comparison.", "required_evidence": ["synthesis", "takeaways"]},
    ],
    "methodology-analysis": [
        {"title": "Problem Setting", "objective": "Define the problem and design constraints.", "required_evidence": ["problem formulation"]},
        {"title": "Method Design", "objective": "Explain the methodological design.", "required_evidence": ["algorithm", "methodology"]},
        {"title": "Optimization and Training", "objective": "Describe optimization or training behavior.", "required_evidence": ["training", "objective"]},
        {"title": "Strengths and Limitations", "objective": "Assess method tradeoffs.", "required_evidence": ["advantages", "limitations"]},
    ],
    "benchmark-analysis": [
        {"title": "Benchmark Context", "objective": "Explain benchmark scope and metrics.", "required_evidence": ["benchmark", "evaluation metrics"]},
        {"title": "Reported Performance", "objective": "Summarize key results and rankings.", "required_evidence": ["results", "benchmark numbers"]},
        {"title": "Interpretation", "objective": "Explain what the reported evidence implies.", "required_evidence": ["analysis", "caveats"]},
    ],
    "survey": [
        {"title": "Introduction", "objective": "Frame the topic and survey scope.", "required_evidence": ["scope", "motivation"]},
        {"title": "Taxonomy", "objective": "Organize the literature into major categories.", "required_evidence": ["themes", "approaches"]},
        {"title": "Representative Methods", "objective": "Summarize representative methods and patterns.", "required_evidence": ["methods", "comparisons"]},
        {"title": "Evaluation Trends", "objective": "Summarize evaluation strategies and evidence trends.", "required_evidence": ["benchmarks", "results"]},
        {"title": "Open Problems", "objective": "Identify unresolved research directions.", "required_evidence": ["limitations", "future work"]},
    ],
}


class OutlinePlanner:
    """Generate structured outlines for research-style reports."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _slug(self, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return normalized or "section"

    def _subsections(self, section_title: str, report_type: str) -> list[OutlineSectionRecord]:
        subsection_titles = {
            "Introduction": ["Scope", "Motivation"],
            "Core Concepts": ["Definitions", "Terminology"],
            "Major Approaches": ["Method Families", "Representative Papers"],
            "Empirical Evidence": ["Benchmarks", "Observed Tradeoffs"],
            "Comparative Analysis": ["Shared Themes", "Key Differences"],
            "Method Design": ["Objective", "Mechanism"],
            "Reported Performance": ["Headline Results", "Caveats"],
            "Taxonomy": ["Categories", "Linking Themes"],
        }.get(section_title, [])
        return [
            OutlineSectionRecord(
                section_id=f"{self._slug(section_title)}-{index:02d}",
                title=title,
                objective=f"Develop the {title.lower()} of {section_title.lower()}.",
                required_evidence=[title.lower(), report_type.replace("-", " ")],
                retrieval_strategy="hybrid_subsection",
            )
            for index, title in enumerate(subsection_titles[: self.config.writer.max_subsections_per_section], start=1)
        ]

    def build(self, topic: str, report_type: str, understanding: QueryUnderstandingResult, *, max_sections: int | None = None) -> OutlinePlan:
        template = REPORT_TEMPLATES.get(report_type, REPORT_TEMPLATES["technical-summary"])
        section_limit = min(max_sections or self.config.writer.max_sections, self.config.writer.max_sections)
        sections: list[OutlineSectionRecord] = []
        for index, item in enumerate(template[:section_limit], start=1):
            title = str(item["title"])
            sections.append(
                OutlineSectionRecord(
                    section_id=f"sec-{index:02d}-{self._slug(title)}",
                    title=title,
                    objective=str(item["objective"]),
                    required_evidence=list(item.get("required_evidence", [])),
                    retrieval_strategy="hybrid_section",
                    subsections=self._subsections(title, report_type),
                    metadata={"query_type": understanding.query_type},
                )
            )
        missing = ["recent empirical evidence", "cross-paper limitations", "open research questions"]
        if understanding.query_type == "comparison":
            missing.insert(0, "direct head-to-head comparison evidence")
        title = f"{topic}: {report_type.replace('-', ' ').title()}"
        return OutlinePlan(
            title=title,
            report_type=report_type,
            topic=topic,
            sections=sections,
            missing_evidence_areas=missing,
            metadata={
                "expanded_terms": understanding.expanded_terms,
                "target_topics": understanding.target_topics,
                "entities": understanding.entities,
            },
        )

