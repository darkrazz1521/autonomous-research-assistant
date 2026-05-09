"""Analytics writers for retrieval workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autonomous_research_assistant_data.models.common import EvaluationReport, RetrievalTrace
from autonomous_research_assistant_data.storage.file_store import append_jsonl, ensure_directory


class RetrievalAnalyticsReporter:
    """Persist retrieval analytics and query traces."""

    def __init__(self, analytics_dir: Path) -> None:
        self.analytics_dir = analytics_dir
        ensure_directory(analytics_dir)

    def write_report(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.analytics_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def append_trace(self, trace: RetrievalTrace) -> Path:
        path = self.analytics_dir / "query_traces.jsonl"
        append_jsonl(path, trace.model_dump(mode="json"))
        return path

    def write_evaluation_report(self, report: EvaluationReport) -> Path:
        return self.write_report(f"evaluation_{report.evaluation_id}", report.model_dump(mode="json"))

