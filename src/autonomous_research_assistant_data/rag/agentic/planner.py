"""Research planning for agentic grounded reasoning."""

from __future__ import annotations

from autonomous_research_assistant_data.models.common import AgenticPlan, AgenticSubtask, QueryUnderstandingResult


class AgenticPlanner:
    """Decompose a research question into bounded retrieval subtasks."""

    def build_plan(self, query: str, understanding: QueryUnderstandingResult) -> AgenticPlan:
        subtasks: list[AgenticSubtask] = []
        query_type = understanding.query_type
        if query_type == "comparison":
            entities = understanding.entities[:2] or ["concept_a", "concept_b"]
            for index, entity in enumerate(entities, start=1):
                subtasks.append(
                    AgenticSubtask(
                        id=f"subtask-{index:02d}",
                        objective=f"Retrieve grounded evidence for {entity}.",
                        retrieval_strategy="hybrid_comparison",
                        priority=index,
                    )
                )
            subtasks.append(
                AgenticSubtask(
                    id=f"subtask-{len(subtasks)+1:02d}",
                    objective="Find direct comparison evidence and tradeoffs across papers.",
                    retrieval_strategy="cross_paper_comparison",
                    priority=3,
                )
            )
        elif query_type in {"contradiction_analysis", "timeline_history", "literature_review"}:
            subtasks.append(
                AgenticSubtask(
                    id="subtask-01",
                    objective="Retrieve broad evidence for the main theme.",
                    retrieval_strategy="broad_synthesis",
                    priority=1,
                )
            )
            subtasks.append(
                AgenticSubtask(
                    id="subtask-02",
                    objective="Retrieve cross-paper supporting and conflicting evidence.",
                    retrieval_strategy="cross_paper_validation",
                    priority=2,
                )
            )
        else:
            subtasks.append(
                AgenticSubtask(
                    id="subtask-01",
                    objective=f"Retrieve direct evidence to answer: {query}",
                    retrieval_strategy="direct_lookup",
                    priority=1,
                )
            )
            subtasks.append(
                AgenticSubtask(
                    id="subtask-02",
                    objective="Retrieve supporting methodological or contextual evidence.",
                    retrieval_strategy="supporting_context",
                    priority=2,
                )
            )
        return AgenticPlan(
            goal=query,
            query_type=query_type,
            subtasks=subtasks,
            metadata={
                "expected_answer_structure": understanding.expected_answer_structure,
                "target_topics": understanding.target_topics,
                "entities": understanding.entities,
            },
        )
