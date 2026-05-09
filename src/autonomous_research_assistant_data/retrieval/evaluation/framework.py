"""Retrieval evaluation framework."""

from __future__ import annotations

import math
from pathlib import Path

from autonomous_research_assistant_data.config import AppConfig
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import EvaluationProbe, EvaluationReport
from autonomous_research_assistant_data.retrieval.api.service import RetrievalApi
from autonomous_research_assistant_data.retrieval.common import load_chunk_records, stable_hash_text
from autonomous_research_assistant_data.storage.file_store import write_json


class RetrievalEvaluationFramework:
    """Evaluate retrieval quality with lightweight scientific probes."""

    def __init__(self, config: AppConfig, api: RetrievalApi) -> None:
        self.config = config
        self.api = api

    def _probe_path(self) -> Path:
        return self.config.retrieval.retrieval_evaluation_dir / "manual_probes.json"

    def build_manual_probes(self, *, limit: int | None = None) -> list[EvaluationProbe]:
        chunk_records = [chunk for _, chunk in load_chunk_records(self.config.pdf_processing.chunks_dir) if chunk.retrieval_quality_score >= 0.65]
        probes: list[EvaluationProbe] = []
        for chunk in chunk_records:
            topics = " ".join(chunk.chunk_topic_signature[:4])
            if topics:
                query = f"{chunk.section_name} {topics}".strip()
            else:
                query = chunk.section_name
            probes.append(
                EvaluationProbe(
                    probe_id=stable_hash_text(f"{chunk.chunk_id}:{query}")[:16],
                    query=query,
                    relevant_chunk_ids=[chunk.chunk_id],
                    relevant_paper_ids=[chunk.paper_id],
                    metadata={"canonical_section_label": chunk.extra.get("canonical_section_label")},
                )
            )
            if chunk.citation_entities:
                citation_query = f"What does {chunk.citation_entities[0]} discuss?"
                probes.append(
                    EvaluationProbe(
                        probe_id=stable_hash_text(f"{chunk.chunk_id}:{citation_query}")[:16],
                        query=citation_query,
                        relevant_chunk_ids=[chunk.chunk_id],
                        relevant_paper_ids=[chunk.paper_id],
                        metadata={"citation_entity": chunk.citation_entities[0]},
                    )
                )
            if limit and len(probes) >= limit:
                break
        write_json(self._probe_path(), {"probes": [probe.model_dump(mode="json") for probe in probes]})
        return probes[:limit] if limit else probes

    def evaluate(self, *, top_k: int | None = None, probe_limit: int | None = None, mode: str = "hybrid", rerank: bool = False) -> EvaluationReport:
        k = top_k or self.config.retrieval.evaluation.default_top_k
        probes = self.build_manual_probes(limit=probe_limit or self.config.retrieval.evaluation.default_probe_count)
        recall_hits = 0
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        citation_grounding: list[float] = []
        latencies: list[float] = []

        for probe in probes:
            trace = self.api.search(probe.query, top_k=k, mode=mode, rerank=rerank, citation_aware=True)
            latencies.append(trace.latency_ms)
            ranked_ids = [item.chunk_id for item in trace.results]
            hit_rank = None
            gains: list[float] = []
            for rank, chunk_id in enumerate(ranked_ids, start=1):
                relevant = 1.0 if chunk_id in probe.relevant_chunk_ids else 0.0
                gains.append(relevant)
                if relevant and hit_rank is None:
                    hit_rank = rank
            if hit_rank is not None:
                recall_hits += 1
                reciprocal_ranks.append(1.0 / hit_rank)
            else:
                reciprocal_ranks.append(0.0)
            dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains[:k]))
            idcg = 1.0
            ndcgs.append(dcg / idcg)
            if "citation_entity" in probe.metadata:
                citation_grounding.append(1.0 if any(item.citation_entities for item in trace.results[:3]) else 0.0)

        report = EvaluationReport(
            evaluation_id=stable_hash_text(f"{utc_now().isoformat()}:{mode}:{rerank}")[:16],
            probe_count=len(probes),
            top_k=k,
            recall_at_k=round(recall_hits / max(len(probes), 1), 4),
            mrr=round(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1), 4),
            ndcg_at_k=round(sum(ndcgs) / max(len(ndcgs), 1), 4),
            citation_grounding_score=round(sum(citation_grounding) / max(len(citation_grounding), 1), 4) if citation_grounding else 0.0,
            latency_ms_mean=round(sum(latencies) / max(len(latencies), 1), 4),
            metadata={"mode": mode, "rerank": rerank, "probe_path": str(self._probe_path())},
        )
        write_json(
            self.config.retrieval.retrieval_evaluation_dir / f"evaluation_{report.evaluation_id}.json",
            report.model_dump(mode="json"),
        )
        return report
