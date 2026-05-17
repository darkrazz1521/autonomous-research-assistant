"""Higher-quality semantic synthesis for scientific report sections."""

from __future__ import annotations

from autonomous_research_assistant_data.models.common import QueryUnderstandingResult, RetrievalResult, WritingSectionPlan
from autonomous_research_assistant_data.writer.citations.citation_manager import CitationManager
from autonomous_research_assistant_data.writer.synthesis.claim_graph import ClaimGraph
from autonomous_research_assistant_data.writer.synthesis.claim_builder import ClaimBuilder
from autonomous_research_assistant_data.writer.synthesis.evidence_merger import EvidenceMerger
from autonomous_research_assistant_data.writer.synthesis.paragraph_constructor import ParagraphConstructor


class SemanticSynthesizer:
    """Produce claim-level, paragraph-level grounded synthesis from retrieval results."""

    def __init__(self) -> None:
        self.merger = EvidenceMerger()
        self.claim_builder = ClaimBuilder()
        self.paragraph_constructor = ParagraphConstructor()
        self.claim_graph = ClaimGraph()

    def synthesize(
        self,
        *,
        topic: str,
        plan: WritingSectionPlan,
        understanding: QueryUnderstandingResult,
        results: list[RetrievalResult],
        citation_manager: CitationManager,
        style_profile: dict[str, object],
        previous_title: str | None,
        prior_claims: list[str] | None = None,
        rhetorical_plan: dict[str, object] | None = None,
    ) -> dict[str, object]:
        clusters = self.merger.cluster(results)
        deduped_clusters = self.merger.deduplicate_claims(clusters)
        claims = self.claim_builder.build(deduped_clusters, plan=plan, understanding=understanding)
        deduped_claims, claim_graph_summary = self.claim_graph.deduplicate(claims, prior_claims=prior_claims)
        paragraphs, traces = self.paragraph_constructor.build(
            topic=topic,
            plan=plan,
            understanding=understanding,
            claims=deduped_claims,
            citation_manager=citation_manager,
            style_profile=style_profile,
            previous_title=previous_title,
            rhetorical_plan=rhetorical_plan,
        )
        cited_results: list[RetrievalResult] = []
        seen: set[str] = set()
        for claim in deduped_claims:
            for result in claim["supporting_results"]:
                assert isinstance(result, RetrievalResult)
                if result.chunk_id in seen:
                    continue
                seen.add(result.chunk_id)
                cited_results.append(result)
        return {
            "paragraphs": paragraphs,
            "claims": deduped_claims,
            "traces": traces,
            "clusters": deduped_clusters,
            "cited_results": cited_results[:8],
            "claim_graph_summary": claim_graph_summary,
            "normalization_reports": [report for cluster in deduped_clusters for report in cluster.get("normalization_reports", [])],
        }
