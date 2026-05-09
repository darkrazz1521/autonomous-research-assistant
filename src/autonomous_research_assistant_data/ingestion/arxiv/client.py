"""arXiv API client and XML parsing."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.retry import retry_async
from autonomous_research_assistant_data.models.common import ArxivPaperRecord

ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivClient:
    """Async client for querying the arXiv Atom API."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.headers = {"User-Agent": config.arxiv.user_agent}

    def _query_string(self) -> str:
        categories = " OR ".join(f"cat:{category}" for category in self.config.arxiv.categories)
        return f"({categories})"

    async def fetch_batch(self, client: httpx.AsyncClient, start: int, max_results: int) -> list[ArxivPaperRecord]:
        params = {
            "search_query": self._query_string(),
            "start": start,
            "max_results": max_results,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
        }

        async def _request() -> httpx.Response:
            response = await client.get(
                self.config.arxiv.base_url,
                params=params,
                headers=self.headers,
                timeout=self.config.arxiv.request_timeout_seconds,
            )
            response.raise_for_status()
            return response

        response = await retry_async(_request, self.config.retry, (httpx.HTTPError,))
        await asyncio.sleep(self.config.arxiv.request_pause_seconds)
        return self._parse_feed(response.text)

    def _parse_feed(self, xml_text: str) -> list[ArxivPaperRecord]:
        root = ET.fromstring(xml_text)
        papers: list[ArxivPaperRecord] = []
        for entry in root.findall("atom:entry", ATOM_NAMESPACE):
            categories = [node.attrib["term"] for node in entry.findall("atom:category", ATOM_NAMESPACE)]
            authors = [
                author.findtext("atom:name", default="", namespaces=ATOM_NAMESPACE)
                for author in entry.findall("atom:author", ATOM_NAMESPACE)
            ]
            paper_id_url = entry.findtext("atom:id", default="", namespaces=ATOM_NAMESPACE).strip()
            paper_id = paper_id_url.rsplit("/", 1)[-1]
            pdf_url = ""
            for link in entry.findall("atom:link", ATOM_NAMESPACE):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib["href"]
                    break
            papers.append(
                ArxivPaperRecord(
                    arxiv_id=paper_id,
                    title=entry.findtext("atom:title", default="", namespaces=ATOM_NAMESPACE).strip(),
                    authors=[author for author in authors if author],
                    abstract=entry.findtext("atom:summary", default="", namespaces=ATOM_NAMESPACE).strip(),
                    categories=categories,
                    published_at=datetime.fromisoformat(
                        entry.findtext("atom:published", default="", namespaces=ATOM_NAMESPACE).replace("Z", "+00:00")
                    ),
                    updated_at=datetime.fromisoformat(
                        entry.findtext("atom:updated", default="", namespaces=ATOM_NAMESPACE).replace("Z", "+00:00")
                    ),
                    pdf_url=pdf_url,
                )
            )
        return papers

    def watermark(self, last_updated_at: str | None) -> datetime:
        if last_updated_at:
            return datetime.fromisoformat(last_updated_at)
        return datetime.now(timezone.utc) - timedelta(days=self.config.arxiv.incremental_lookback_days)

