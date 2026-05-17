"""Persistent conversation session memory."""

from __future__ import annotations

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import ConversationTurn, RAGAnswer, ResearchSessionRecord
from autonomous_research_assistant_data.storage.file_store import read_json, write_json


class SessionMemoryStore:
    """Load and persist conversation sessions for follow-up aware RAG."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _path(self, conversation_id: str):
        return self.config.rag.research_sessions_dir / f"{conversation_id}.json"

    def load(self, conversation_id: str) -> ResearchSessionRecord:
        payload = read_json(self._path(conversation_id), default={})
        if payload:
            return ResearchSessionRecord.model_validate(payload)
        return ResearchSessionRecord(conversation_id=conversation_id, updated_at=utc_now())

    def save(self, session: ResearchSessionRecord) -> None:
        write_json(self._path(session.conversation_id), session.model_dump(mode="json"))

    def append_answer(self, conversation_id: str, answer: RAGAnswer) -> ResearchSessionRecord:
        session = self.load(conversation_id)
        cited_papers = list(dict.fromkeys(record.paper_id for record in answer.citations))
        topics = list(dict.fromkeys(answer.retrieval_metadata.get("active_topics", []))) if isinstance(answer.retrieval_metadata.get("active_topics", []), list) else []
        turn = ConversationTurn(
            turn_id=f"{conversation_id}-t{len(session.turns):04d}",
            query=answer.query,
            answer=answer.answer,
            cited_paper_ids=cited_papers,
            active_topics=topics,
            unresolved_questions=answer.grounding_report.unsupported_claims if answer.grounding_report else [],
            created_at=utc_now(),
            metadata={"confidence_score": answer.confidence_score},
        )
        session.turns.append(turn)
        session.discussed_papers = list(dict.fromkeys(session.discussed_papers + cited_papers))
        session.active_research_topics = list(dict.fromkeys(session.active_research_topics + topics))[: self.config.rag.memory.max_active_topics]
        session.unresolved_questions = list(dict.fromkeys(session.unresolved_questions + turn.unresolved_questions))
        session.citation_reuse = list(dict.fromkeys(session.citation_reuse + [record.citation_label for record in answer.citations]))
        session.updated_at = utc_now()
        if len(session.turns) > self.config.rag.memory.max_history_items:
            session.turns = session.turns[-self.config.rag.memory.max_history_items :]
        self.save(session)
        return session
