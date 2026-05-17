"""Rhetorical planning for section-specific scientific writing behavior."""

from __future__ import annotations

from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, WritingSectionPlan


class RhetoricalPlanner:
    """Assign rhetorical intent and paragraph roles per section."""

    def classify(self, section: WritingSectionPlan, understanding: QueryUnderstandingResult) -> str:
        lowered = section.title.lower()
        if "introduction" in lowered or "background" in lowered or "overview" in lowered:
            return "introduction"
        if "method" in lowered or "approach" in lowered or "algorithm" in lowered:
            return "methods"
        if understanding.query_type == "comparison" or "comparison" in lowered or "versus" in lowered:
            return "comparison"
        if "limitation" in lowered or "open" in lowered or "risk" in lowered:
            return "limitations"
        if "conclusion" in lowered or "future" in lowered:
            return "conclusion"
        return "analysis"

    def plan(self, section: WritingSectionPlan, understanding: QueryUnderstandingResult) -> dict[str, object]:
        section_type = self.classify(section, understanding)
        if section_type == "introduction":
            roles = ["framing", "motivation", "problem_setup"]
        elif section_type == "methods":
            roles = ["technical_grounding", "procedure", "mechanism"]
        elif section_type == "comparison":
            roles = ["shared_context", "difference_analysis", "tradeoff_synthesis"]
        elif section_type == "limitations":
            roles = ["boundary_condition", "uncertainty", "caution"]
        elif section_type == "conclusion":
            roles = ["synthesis", "takeaway", "future_direction"]
        else:
            roles = ["claim", "evidence", "interpretation"]
        return {
            "section_type": section_type,
            "paragraph_roles": roles,
            "discourse_policy": {
                "prefer_contrastive_language": section_type == "comparison",
                "prefer_cautionary_language": section_type in {"limitations", "conclusion"},
                "prefer_procedural_language": section_type == "methods",
            },
            "rhetorical_sequence": roles,
        }
