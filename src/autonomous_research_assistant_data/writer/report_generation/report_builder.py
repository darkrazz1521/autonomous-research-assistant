"""Persist export-ready report objects."""

from __future__ import annotations

import re
from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.models.common import ResearchReport
from autonomous_research_assistant_data.storage.file_store import write_json


class ReportBuilder:
    """Write report artifacts in markdown, JSON, or structured form."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "report"

    def _markdown(self, report: ResearchReport) -> str:
        blocks = [
            f"# {report.title}",
            "",
            "## Abstract",
            report.abstract,
            "",
            "## Introduction",
            report.introduction,
            "",
        ]
        for section in report.sections:
            blocks.extend([f"## {section.title}", section.content, ""])
        blocks.extend(["## Conclusion", report.conclusion, "", "## Bibliography"])
        blocks.extend(f"- {entry}" for entry in report.bibliography)
        return "\n".join(blocks).strip() + "\n"

    def persist(self, report: ResearchReport) -> dict[str, str]:
        base = self.config.writer.generated_reports_dir / f"{self._slug(report.title)}-{report.report_id}"
        paths: dict[str, str] = {}
        markdown_path = Path(f"{base}.md")
        json_path = Path(f"{base}.json")
        structured_path = Path(f"{base}.report.json")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(self._markdown(report), encoding="utf-8")
        write_json(json_path, report.model_dump(mode="json"))
        write_json(
            structured_path,
            {
                "report_id": report.report_id,
                "title": report.title,
                "topic": report.topic,
                "report_type": report.report_type,
                "sections": [section.model_dump(mode="json") for section in report.sections],
                "bibliography": report.bibliography,
                "metadata": report.metadata,
            },
        )
        paths["markdown"] = str(markdown_path)
        paths["json"] = str(json_path)
        paths["structured"] = str(structured_path)
        return paths
