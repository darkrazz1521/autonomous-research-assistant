"""Shared data models used across ingestion and PDF processing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArxivPaperRecord(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_at: datetime
    updated_at: datetime
    pdf_url: str
    pdf_path: Path | None = None
    metadata_path: Path | None = None
    source: str = "arxiv"
    download_timestamp: datetime | None = None


class DatasetArtifactRecord(BaseModel):
    dataset_name: str
    source_type: str = "huggingface"
    source_id: str
    config_name: str | None = None
    split: str
    num_rows: int
    storage_path: Path | None = None
    fingerprint: str | None = None
    downloaded_at: datetime
    loader_strategy: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ManifestEntry(BaseModel):
    entry_id: str
    source: str
    status: str
    created_at: datetime
    updated_at: datetime
    payload: dict[str, Any]


class ExtractedPage(BaseModel):
    page_number: int
    text: str
    lines: list[str]
    blocks: list[str] = Field(default_factory=list)
    block_metadata: list[dict[str, Any]] = Field(default_factory=list)


class ExcludedArtifact(BaseModel):
    artifact_id: str
    artifact_type: str
    text: str
    page_number: int | None = None
    confidence: float = 0.0


class ReferenceEntry(BaseModel):
    reference_id: str
    text: str
    page_number: int | None = None
    citation_key: str | None = None


class CitationSpan(BaseModel):
    citation_id: str
    text: str
    start_offset: int
    end_offset: int
    citation_type: str
    entities: list[str] = Field(default_factory=list)
    year: int | None = None
    normalized_key: str | None = None


class EquationBlock(BaseModel):
    equation_id: str
    paper_id: str
    arxiv_id: str
    text: str
    normalized_text: str
    page_start: int | None = None
    page_end: int | None = None
    paragraph_ids: list[str] = Field(default_factory=list)
    is_multiline: bool = False
    operator_count: int = 0
    confidence: float = 0.0
    chunk_guard: bool = True
    equation_integrity_score: float = 0.0
    equation_type: str = "display"


class HeadingRecord(BaseModel):
    heading_id: str
    paper_id: str
    arxiv_id: str
    text: str
    normalized_heading: str
    page_number: int | None = None
    paragraph_id: str | None = None
    heading_confidence: float = 0.0
    heading_type: str = "unknown"
    hierarchy_depth: int = 1
    canonical_section_label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LayoutRegion(BaseModel):
    region_id: str
    paper_id: str
    arxiv_id: str
    page_number: int
    region_type: str
    text: str
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DedupGroup(BaseModel):
    group_id: str
    paper_id: str
    arxiv_id: str
    paragraph_ids: list[str] = Field(default_factory=list)
    canonical_paragraph_id: str | None = None
    dedup_confidence: float = 0.0
    overlap_cluster: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FrontMatterRecord(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    arxiv_metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedDocument(BaseModel):
    paper_id: str
    arxiv_id: str
    source_pdf: Path
    extraction_backend: str
    extraction_fallback_used: bool = False
    extraction_quality_score: float
    title: str | None = None
    abstract: str | None = None
    references: list[str] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)
    pages: list[ExtractedPage]
    extracted_text: str
    page_count: int
    processed_at: datetime
    extra: dict[str, Any] = Field(default_factory=dict)


class CleanParagraph(BaseModel):
    paragraph_id: str
    text: str
    page_number: int
    page_end: int | None = None
    line_span: tuple[int, int] | None = None
    is_equation: bool = False
    contains_citation: bool = False
    artifact_type: str | None = None
    section_hint: str | None = None
    semantic_density_score: float = 0.0
    repair_confidence: float = 1.0
    structural_role: str | None = None
    equation_block_id: str | None = None
    equation_integrity_score: float = 0.0
    equation_type: str | None = None
    citation_spans: list[CitationSpan] = Field(default_factory=list)
    noise_classifications: list[str] = Field(default_factory=list)
    heading_confidence: float = 0.0
    heading_type: str | None = None
    hierarchy_depth: int = 0
    normalized_heading: str | None = None
    dedup_confidence: float = 0.0
    duplicate_group_id: str | None = None
    overlap_cluster: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionRecord(BaseModel):
    section_id: str
    paper_id: str
    arxiv_id: str
    section_index: int
    section_name: str
    normalized_section_name: str
    canonical_section_label: str
    heading_raw: str | None = None
    confidence: float
    level: int = 1
    parent_section_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    contains_references: bool = False
    paragraphs: list[CleanParagraph] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    paper_id: str
    arxiv_id: str
    source_pdf: Path
    section_name: str
    section_index: int
    chunk_index: int
    chunk_text: str
    token_count_estimate: int
    page_range: tuple[int | None, int | None]
    processing_timestamp: datetime
    paragraph_ids: list[str] = Field(default_factory=list)
    semantic_hash: str | None = None
    parent_section_id: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    chunk_topic_signature: list[str] = Field(default_factory=list)
    contains_equation: bool = False
    contains_citation: bool = False
    citation_spans: list[CitationSpan] = Field(default_factory=list)
    citation_entities: list[str] = Field(default_factory=list)
    coherence_score: float = 0.0
    noise_score: float = 0.0
    structure_score: float = 0.0
    semantic_density_score: float = 0.0
    retrieval_quality_score: float = 0.0
    citation_density: float = 0.0
    equation_density: float = 0.0
    scientific_complexity_score: float = 0.0
    repair_confidence: float = 1.0
    structural_integrity_score: float = 0.0
    transition_quality_score: float = 0.0
    semantic_boundary_score: float = 0.0
    narrative_continuity_score: float = 0.0
    noise_classifications: list[str] = Field(default_factory=list)
    corruption_categories: list[str] = Field(default_factory=list)
    repair_recommendations: list[str] = Field(default_factory=list)
    structural_anomaly_score: float = 0.0
    flagged_for_review: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
    retrieval_excluded: bool = False
    numeric_ratio: float = 0.0
    alphabetic_ratio: float = 0.0
    language_entropy: float = 0.0
    table_probability: float = 0.0
    benchmark_probability: float = 0.0
    duplicate_line_ratio: float = 0.0
    average_sentence_length: float = 0.0
    malformed_structure_score: float = 0.0


class ProcessingReport(BaseModel):
    paper_id: str
    arxiv_id: str
    source_pdf: Path
    extraction_backend: str
    extraction_quality_score: float
    section_count: int
    chunk_count: int
    empty_chunk_count: int
    duplicate_chunk_count: int
    tiny_chunk_count: int
    oversized_chunk_count: int
    overlap_duplicate_ratio: float = 0.0
    equation_fragmentation_count: int = 0
    reference_leakage_count: int = 0
    incoherent_chunk_count: int = 0
    low_quality_chunk_count: int = 0
    noise_ratio_max: float
    ocr_corruption_count: int = 0
    equation_corruption_count: int = 0
    table_bleed_count: int = 0
    caption_contamination_count: int = 0
    duplicated_section_count: int = 0
    malformed_unicode_count: int = 0
    incomplete_reference_count: int = 0
    chunk_fragmentation_count: int = 0
    repair_effectiveness_score: float = 0.0
    malformed_heading_count: int = 0
    duplicate_paragraph_count: int = 0
    column_merge_corruption_count: int = 0
    layout_contamination_count: int = 0
    figure_leakage_count: int = 0
    chunk_discontinuity_count: int = 0
    status: str
    processed_at: datetime
    extra: dict[str, Any] = Field(default_factory=dict)


class EmbeddingRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    embedding_id: str
    chunk_id: str
    paper_id: str
    arxiv_id: str
    model_name: str
    vector_dim: int
    vector_norm: float = 0.0
    normalized: bool = True
    namespace: str = "default"
    chunk_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime


class VectorIndexRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    index_id: str
    namespace: str
    backend: str
    model_name: str
    vector_dim: int
    document_count: int = 0
    index_path: Path | None = None
    metadata_path: Path | None = None
    created_at: datetime
    updated_at: datetime
    extra: dict[str, Any] = Field(default_factory=dict)


class NeighborRecord(BaseModel):
    source_chunk_id: str
    target_chunk_id: str
    relation_type: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    paper_id: str
    arxiv_id: str
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    raw_vector_score: float | None = None
    raw_sparse_score: float | None = None
    rerank_score: float | None = None
    section_weight: float = 1.0
    final_retrieval_score: float | None = None
    final_score_breakdown: dict[str, Any] = Field(default_factory=dict)
    citation_boost: float = 0.0
    section_boost: float = 0.0
    rank: int = 0
    chunk_text: str
    section_name: str
    canonical_section_label: str | None = None
    citations: list[str] = Field(default_factory=list)
    citation_entities: list[str] = Field(default_factory=list)
    neighboring_chunk_ids: list[str] = Field(default_factory=list)
    primary_chunk: dict[str, Any] = Field(default_factory=dict)
    context_before: list[dict[str, Any]] = Field(default_factory=list)
    context_after: list[dict[str, Any]] = Field(default_factory=list)
    merged_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    query: str
    mode: str
    namespace: str = "default"
    top_k: int
    latency_ms: float = 0.0
    dense_latency_ms: float = 0.0
    sparse_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    results: list[RetrievalResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationProbe(BaseModel):
    probe_id: str
    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_paper_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    evaluation_id: str
    probe_count: int
    top_k: int
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    citation_grounding_score: float = 0.0
    latency_ms_mean: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
