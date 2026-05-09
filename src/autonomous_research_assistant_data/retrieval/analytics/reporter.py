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

    def write_chunk_quality_summary(self, rows: list[dict[str, Any]]) -> Path:
        by_paper: dict[str, int] = {}
        excluded = 0
        for row in rows:
            paper_id = str(row.get("paper_id", "unknown"))
            by_paper[paper_id] = by_paper.get(paper_id, 0) + 1
            excluded += int(bool(row.get("retrieval_excluded")))
        payload = {
            "total_chunks": len(rows),
            "excluded_chunks": excluded,
            "average_chunk_quality": round(sum(float(row.get("retrieval_quality_score", 0.0)) for row in rows) / max(len(rows), 1), 6),
            "average_chunk_noise": round(sum(float(row.get("retrieval_noise_score", 0.0)) for row in rows) / max(len(rows), 1), 6),
            "top_noisy_papers": sorted(by_paper.items(), key=lambda item: item[1], reverse=True)[:10],
        }
        return self.write_report("chunk_quality_summary", payload)
