"""Persistent writing memory for section-level report generation."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import OutlinePlan, SectionDraft, WritingMemoryEntry, WritingPlan, WriterSessionRecord
from autonomous_research_assistant_data.storage.file_store import read_json, write_json


class WritingMemoryStore:
    """Persist and update writer sessions."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _path(self, session_id: str):
        return self.config.writer.writer_sessions_dir / f"{session_id}.json"

    def create(self, *, session_id: str, topic: str, report_type: str, title: str) -> WriterSessionRecord:
        session = WriterSessionRecord(session_id=session_id, topic=topic, report_type=report_type, title=title, updated_at=utc_now())
        self.save(session)
        return session

    def load(self, session_id: str) -> WriterSessionRecord | None:
        payload = read_json(self._path(session_id), default={})
        if not payload:
            return None
        return WriterSessionRecord.model_validate(payload)

    def save(self, session: WriterSessionRecord) -> None:
        session.updated_at = utc_now()
        write_json(self._path(session.session_id), session.model_dump(mode="json"))

    def attach_outline(self, session: WriterSessionRecord, outline: OutlinePlan) -> WriterSessionRecord:
        session.outline = outline
        self.save(session)
        return session

    def attach_plan(self, session: WriterSessionRecord, plan: WritingPlan) -> WriterSessionRecord:
        session.writing_plan = plan
        self.save(session)
        return session

    def add_section(self, session: WriterSessionRecord, draft: SectionDraft) -> WriterSessionRecord:
        session.sections = [item for item in session.sections if item.section_id != draft.section_id] + [draft]
        entry = WritingMemoryEntry(
            section_id=draft.section_id,
            title=draft.title,
            summary=draft.summary,
            used_citation_labels=[item.citation_label for item in draft.citations],
            writing_decisions=list(draft.metadata.get("writing_decisions", [])),
            terminology=draft.terminology,
            repeated_concepts=list(draft.metadata.get("repeated_concepts", [])),
            unresolved_gaps=draft.unresolved_gaps,
        )
        session.writing_memory = [item for item in session.writing_memory if item.section_id != draft.section_id] + [entry]
        session.used_citations = list(dict.fromkeys(session.used_citations + [item.citation_label for item in draft.citations]))
        session.unresolved_gaps = list(dict.fromkeys(session.unresolved_gaps + draft.unresolved_gaps))
        for term in draft.terminology:
            session.terminology_map.setdefault(term.lower(), term)
        self.save(session)
        return session

    def latest_summary(self, session: WriterSessionRecord) -> str:
        if not session.sections:
            return ""
        return session.sections[-1].summary

    def prior_titles(self, session: WriterSessionRecord) -> list[str]:
        return [section.title for section in session.sections]

