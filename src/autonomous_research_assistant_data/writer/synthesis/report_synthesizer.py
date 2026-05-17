"""Merge section drafts into coherent long-form reports."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import OutlinePlan, ResearchReport, SectionDraft, WriterSessionRecord
from autonomous_research_assistant_data.writer.style.style_controller import StyleController


class ReportSynthesizer:
    """Compose section drafts into a coherent report object."""

    def __init__(self, config: AppConfig, style_controller: StyleController | None = None) -> None:
        self.config = config
        self.style_controller = style_controller or StyleController()

    def _abstract(self, topic: str, sections: list[SectionDraft]) -> str:
        summaries = [section.summary for section in sections[:3] if section.summary]
        if not summaries:
            return f"This report synthesizes grounded evidence on {topic}."
        return " ".join(summaries)[:650].strip()

    def _introduction(self, topic: str, report_type: str, sections: list[SectionDraft]) -> str:
        if sections:
            return (
                f"This {report_type.replace('-', ' ')} examines {topic} using the retrieved scientific corpus. "
                f"The report organizes the evidence around {', '.join(section.title.lower() for section in sections[:3])}."
            )
        return f"This report examines {topic} using grounded retrieval and citation-aware synthesis."

    def _conclusion(self, sections: list[SectionDraft], session: WriterSessionRecord | None) -> str:
        summaries = [section.summary for section in sections[-2:] if section.summary]
        unresolved = session.unresolved_gaps[:3] if session else []
        body = " ".join(summaries) if summaries else "The report synthesizes the strongest retrieved evidence while retaining uncertainty where the corpus is incomplete."
        if unresolved:
            body += f" Remaining open issues include {', '.join(unresolved)}."
        return body.strip()

    def _content(self, outline: OutlinePlan | None, sections: list[SectionDraft]) -> str:
        blocks: list[str] = []
        if outline:
            blocks.append(f"# {outline.title}")
        for section in sections:
            blocks.append(self.style_controller.section_heading(section.title, level=2))
            blocks.append(section.content.strip())
        return self.style_controller.normalize_text("\n\n".join(blocks))

    def synthesize(
        self,
        *,
        report_id: str,
        session_id: str,
        topic: str,
        report_type: str,
        style: str,
        citation_style: str,
        export_format: str,
        outline: OutlinePlan | None,
        sections: list[SectionDraft],
        bibliography: list[str],
        session: WriterSessionRecord | None = None,
    ) -> ResearchReport:
        content = self._content(outline, sections)
        introduction = self._introduction(topic, report_type, sections)
        conclusion = self._conclusion(sections, session)
        abstract = self._abstract(topic, sections)
        citations = []
        for section in sections:
            citations.extend(section.citations)
        deduped_citations = []
        seen = set()
        for citation in citations:
            key = (citation.paper_id, citation.section_name, citation.chunk_id, citation.citation_label)
            if key in seen:
                continue
            seen.add(key)
            deduped_citations.append(citation)
        from autonomous_research_assistant_data.core.time import utc_now

        return ResearchReport(
            report_id=report_id,
            session_id=session_id,
            topic=topic,
            report_type=report_type,
            title=outline.title if outline else topic,
            style=style,
            citation_style=citation_style,
            export_format=export_format,
            outline=outline,
            sections=sections,
            introduction=introduction,
            conclusion=conclusion,
            abstract=abstract,
            content=content,
            bibliography=bibliography,
            citations=deduped_citations,
            generated_at=utc_now(),
            metadata={
                "section_count": len(sections),
                "repeated_idea_candidates": [section.title for section in sections if section.answer_quality_report and section.answer_quality_report.redundancy_score > 0.4],
            },
        )
