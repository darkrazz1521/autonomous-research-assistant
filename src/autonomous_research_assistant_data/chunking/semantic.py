"""Elite section-aware semantic chunking for scientific text."""

from __future__ import annotations

import re

from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.models.common import ChunkRecord, FrontMatterRecord, SectionRecord
from autonomous_research_assistant_data.processing.citations.parser import CitationParser
from autonomous_research_assistant_data.processing.utils import (
    estimate_token_count,
    semantic_density_score,
    stable_text_hash,
    topic_signature,
)


class SemanticChunker:
    """Create adaptive chunks with minimized overlap and equation-aware grouping."""

    EQUATION_PATTERN = re.compile(r"(=|\\sum|\\prod|\\int|\\frac|Σ|Π|∂|∇|θ|λ|β|γ|≤|≥|≈)")
    LIST_PATTERN = re.compile(r"^(\d+\.|\([a-z]\)|[-*])\s+")
    THEOREM_PATTERN = re.compile(r"^(theorem|lemma|proof|proposition|corollary)\b", re.IGNORECASE)

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_tokens: int,
        max_chunk_tokens: int,
        min_overlap_paragraphs: int = 1,
        max_overlap_paragraphs: int = 2,
        abstract_chunk_max_tokens: int = 220,
        repair_overlap_buffer_tokens: int = 140,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.min_overlap_paragraphs = min_overlap_paragraphs
        self.max_overlap_paragraphs = max_overlap_paragraphs
        self.abstract_chunk_max_tokens = abstract_chunk_max_tokens
        self.repair_overlap_buffer_tokens = repair_overlap_buffer_tokens
        self.citation_parser = CitationParser()

    def _target_chunk_size(self, section: SectionRecord) -> int:
        if section.canonical_section_label in {"abstract", "conclusion"}:
            return min(self.chunk_size, self.abstract_chunk_max_tokens if section.canonical_section_label == "abstract" else 320)
        if section.canonical_section_label in {"methodology", "preliminaries"}:
            return min(self.max_chunk_tokens, self.chunk_size + 120)
        if section.canonical_section_label in {"results", "discussion", "related_work", "experiments"}:
            return min(self.max_chunk_tokens, self.chunk_size + 80)
        return self.chunk_size

    def _paragraph_role(self, paragraph) -> str:
        return paragraph.structural_role or (
            "equation_block"
            if paragraph.is_equation
            else "list_item"
            if self.LIST_PATTERN.match(paragraph.text)
            else "theorem_block"
            if self.THEOREM_PATTERN.match(paragraph.text)
            else "paragraph"
        )

    def _is_semantic_continuation(self, left_text: str, right_text: str) -> bool:
        if not left_text or not right_text:
            return False
        if left_text.endswith("-") and right_text[:1].islower():
            return True
        if not re.search(r"[.!?]\)?[\"']?$", left_text) and right_text[:1].islower():
            return True
        if re.search(r"(Figure|Table|Algorithm)\s+\d+[A-Za-z]?$", left_text) and right_text[:1].isupper():
            return True
        return False

    def _should_split(self, section: SectionRecord, current_paragraphs: list, next_paragraph) -> bool:
        if not current_paragraphs:
            return False
        current_text = "\n\n".join(item.text for item in current_paragraphs)
        current_tokens = estimate_token_count(current_text)
        next_tokens = estimate_token_count(next_paragraph.text)
        target_size = self._target_chunk_size(section)
        last = current_paragraphs[-1]
        last_role = self._paragraph_role(last)
        next_role = self._paragraph_role(next_paragraph)

        if current_tokens >= self.max_chunk_tokens:
            return True
        if last_role in {"equation_block", "theorem_block", "list_item", "table_block"} and current_tokens < self.max_chunk_tokens:
            return False
        if next_role in {"equation_block", "theorem_block", "list_item", "table_block"} and current_tokens < target_size:
            return False
        if self._is_semantic_continuation(last.text, next_paragraph.text):
            return False
        if current_tokens + next_tokens > target_size and current_tokens >= self.min_chunk_tokens:
            return True
        if re.search(r"[.!?]\)?[\"']?$", last.text) and next_paragraph.text[:1].isupper() and current_tokens >= int(target_size * 0.85):
            return True
        return False

    def _compute_scores(self, text: str, section: SectionRecord, contains_equation: bool, citation_density: float, repair_confidence: float) -> tuple[float, float, float, float, float, float]:
        density = semantic_density_score(text)
        tokens = estimate_token_count(text)
        sentence_endings = max(1, len(re.findall(r"[.!?]", text)))
        avg_sentence_len = tokens / sentence_endings
        coherence = max(0.0, min(1.0, 1.0 - abs(avg_sentence_len - 24) / 32))
        noise = min(1.0, len(re.findall(r"[^\w\s\.,;:\-\(\)\[\]/%\\=+*<>≤≥≈ΣΠ∂∇θλβγ]", text)) / max(len(text), 1) * 4)
        structure = 0.45 + (0.18 if section.canonical_section_label not in {"front_matter", "references"} else 0.0)
        structure += 0.14 if citation_density > 0 else 0.0
        structure += 0.12 if contains_equation else 0.0
        structure += 0.08 if repair_confidence >= 0.85 else 0.0
        structure = min(1.0, structure)
        scientific_complexity = min(1.0, density * 0.45 + citation_density * 0.2 + (0.2 if contains_equation else 0.0) + min(tokens / 700, 0.15))
        retrieval = max(0.0, min(1.0, (coherence * 0.24) + ((1 - noise) * 0.24) + (structure * 0.18) + (density * 0.18) + (scientific_complexity * 0.16)))
        structural_integrity = max(0.0, min(1.0, (repair_confidence * 0.45) + ((1 - noise) * 0.3) + (coherence * 0.25)))
        return round(coherence, 4), round(noise, 4), round(structure, 4), round(density, 4), round(retrieval, 4), round(structural_integrity, 4)

    def _transition_scores(self, paragraphs: list) -> tuple[float, float, float]:
        if len(paragraphs) <= 1:
            return 0.92, 0.9, 0.94
        continuity_hits = 0
        strong_boundaries = 0
        for left, right in zip(paragraphs, paragraphs[1:]):
            if self._is_semantic_continuation(left.text, right.text):
                continuity_hits += 1
            if re.search(r"[.!?]\)?[\"']?$", left.text) and right.text[:1].isupper():
                strong_boundaries += 1
        transitions = len(paragraphs) - 1
        transition_quality = min(1.0, 0.55 + strong_boundaries / max(transitions, 1) * 0.35)
        semantic_boundary = min(1.0, 0.45 + strong_boundaries / max(transitions, 1) * 0.4)
        narrative_continuity = min(1.0, 0.5 + continuity_hits / max(transitions, 1) * 0.35 + (0.1 if paragraphs[0].section_hint != "heading" else 0.0))
        return round(transition_quality, 4), round(semantic_boundary, 4), round(narrative_continuity, 4)

    def _finalize_chunk(self, chunks: list[ChunkRecord], paper_id: str, arxiv_id: str, source_pdf, section: SectionRecord, chunk_index: int, paragraphs: list) -> ChunkRecord | None:
        chunk_text = "\n\n".join(item.text for item in paragraphs).strip()
        if not chunk_text:
            return None

        citation_data = self.citation_parser.parse(chunk_text, prefix=f"{paper_id}-c{chunk_index:05d}")
        contains_equation = any(item.is_equation for item in paragraphs) or bool(self.EQUATION_PATTERN.search(chunk_text))
        repair_confidence = round(sum(item.repair_confidence for item in paragraphs) / max(len(paragraphs), 1), 4)
        equation_density = round(sum(1 for item in paragraphs if item.is_equation) / max(len(paragraphs), 1), 4)
        coherence, noise, structure, density, retrieval, structural_integrity = self._compute_scores(
            chunk_text,
            section,
            contains_equation,
            float(citation_data["citation_density"]),
            repair_confidence,
        )
        scientific_complexity = round(
            min(
                1.0,
                density * 0.4
                + float(citation_data["citation_density"]) * 0.2
                + equation_density * 0.22
                + min(estimate_token_count(chunk_text) / self.max_chunk_tokens, 1.0) * 0.18,
            ),
            4,
        )
        transition_quality, semantic_boundary, narrative_continuity = self._transition_scores(paragraphs)
        noise_labels = sorted({label for item in paragraphs for label in item.noise_classifications})
        corruption_categories = sorted({label for label in noise_labels if label in {"table_bleed", "caption_candidate", "multi_column_risk", "ocr_artifact", "figure_region"}})
        structural_anomaly = round(
            min(
                1.0,
                noise * 0.45
                + (1 - structural_integrity) * 0.25
                + max(0.0, 0.7 - transition_quality) * 0.2
                + max(0.0, 0.7 - narrative_continuity) * 0.1,
            ),
            4,
        )
        repair_recommendations: list[str] = []
        if "multi_column_risk" in corruption_categories:
            repair_recommendations.append("Re-run with column reconstruction enabled.")
        if "table_bleed" in corruption_categories:
            repair_recommendations.append("Inspect table isolation heuristics for numeric region suppression.")
        if "caption_candidate" in corruption_categories or "figure_region" in corruption_categories:
            repair_recommendations.append("Review figure and caption suppression around this chunk.")
        if contains_equation and equation_density > 0.4:
            repair_recommendations.append("Inspect equation boundaries for display-math preservation.")
        chunk = ChunkRecord(
            chunk_id=f"{paper_id}-c{chunk_index:05d}",
            paper_id=paper_id,
            arxiv_id=arxiv_id,
            source_pdf=source_pdf,
            section_name=section.section_name,
            section_index=section.section_index,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            token_count_estimate=estimate_token_count(chunk_text),
            page_range=(min(item.page_number for item in paragraphs), max(item.page_end or item.page_number for item in paragraphs)),
            processing_timestamp=utc_now(),
            paragraph_ids=[item.paragraph_id for item in paragraphs],
            semantic_hash=stable_text_hash(chunk_text),
            parent_section_id=section.section_id,
            contains_equation=contains_equation,
            contains_citation=bool(citation_data["citation_spans"]),
            citation_spans=citation_data["citation_spans"],
            citation_entities=citation_data["citation_entities"],
            chunk_topic_signature=topic_signature(chunk_text),
            coherence_score=coherence,
            noise_score=noise,
            structure_score=structure,
            semantic_density_score=density,
            retrieval_quality_score=retrieval,
            citation_density=float(citation_data["citation_density"]),
            equation_density=equation_density,
            scientific_complexity_score=scientific_complexity,
            repair_confidence=repair_confidence,
            structural_integrity_score=structural_integrity,
            transition_quality_score=transition_quality,
            semantic_boundary_score=semantic_boundary,
            narrative_continuity_score=narrative_continuity,
            noise_classifications=noise_labels,
            corruption_categories=corruption_categories,
            repair_recommendations=repair_recommendations,
            structural_anomaly_score=structural_anomaly,
            flagged_for_review=retrieval < 0.52 or noise > 0.28 or structural_integrity < 0.62 or structural_anomaly > 0.4,
            extra={"canonical_section_label": section.canonical_section_label, "citation_offsets": citation_data["citation_offsets"]},
        )
        if chunks:
            chunk.previous_chunk_id = chunks[-1].chunk_id
            chunks[-1].next_chunk_id = chunk.chunk_id
        return chunk

    def _carry_overlap(self, paragraphs: list) -> list:
        if not paragraphs:
            return []
        overlap: list = []
        current_tokens = 0
        required_roles = {"equation_block", "theorem_block", "list_item", "table_block"}
        for paragraph in reversed(paragraphs):
            overlap.insert(0, paragraph)
            current_tokens += estimate_token_count(paragraph.text)
            if len(overlap) >= self.max_overlap_paragraphs:
                break
            if len(overlap) >= self.min_overlap_paragraphs and current_tokens >= self.chunk_overlap:
                break
            if self._paragraph_role(paragraph) in required_roles and current_tokens < self.repair_overlap_buffer_tokens:
                continue
        return overlap

    def build_abstract_chunk(self, paper_id: str, arxiv_id: str, source_pdf, front_matter: FrontMatterRecord) -> ChunkRecord | None:
        abstract = (front_matter.abstract or "").strip()
        if not abstract:
            return None
        trimmed = abstract
        while estimate_token_count(trimmed) > self.abstract_chunk_max_tokens and " " in trimmed:
            trimmed = " ".join(trimmed.split()[:-10])
        section = SectionRecord(
            section_id=f"{paper_id}-abstract",
            paper_id=paper_id,
            arxiv_id=arxiv_id,
            section_index=0,
            section_name="Abstract",
            normalized_section_name="abstract",
            canonical_section_label="abstract",
            confidence=1.0,
        )
        pseudo_paragraph = type(
            "PseudoParagraph",
            (),
            {
                "text": trimmed,
                "page_number": 1,
                "page_end": 1,
                "paragraph_id": f"{paper_id}-abstract-p0000",
                "is_equation": False,
                "repair_confidence": 1.0,
                "noise_classifications": [],
                "section_hint": None,
            },
        )()
        chunk = self._finalize_chunk([], paper_id, arxiv_id, source_pdf, section, 0, [pseudo_paragraph])
        if chunk is not None:
            chunk.retrieval_quality_score = max(chunk.retrieval_quality_score, 0.75)
            chunk.structure_score = max(chunk.structure_score, 0.8)
            chunk.extra["priority"] = "high"
        return chunk

    def chunk_sections(self, paper_id: str, arxiv_id: str, source_pdf, sections: list[SectionRecord], front_matter: FrontMatterRecord | None = None) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        next_chunk_index = 0
        abstract_chunk = self.build_abstract_chunk(paper_id, arxiv_id, source_pdf, front_matter) if front_matter else None
        if abstract_chunk is not None:
            chunks.append(abstract_chunk)
            next_chunk_index = 1

        for section in sections:
            if section.canonical_section_label in {"references", "front_matter"}:
                continue
            current_paragraphs: list = []
            current_ids: set[str] = set()
            for paragraph in section.paragraphs:
                if paragraph.artifact_type in {"caption", "email", "affiliation", "watermark", "link"}:
                    continue
                if paragraph.metadata.get("isolated_region_type") in {"figure", "table"}:
                    continue
                if self._should_split(section, current_paragraphs, paragraph):
                    chunk = self._finalize_chunk(chunks, paper_id, arxiv_id, source_pdf, section, next_chunk_index, current_paragraphs)
                    if chunk is not None:
                        chunks.append(chunk)
                        next_chunk_index += 1
                    current_paragraphs = self._carry_overlap(current_paragraphs)
                    current_ids = {item.paragraph_id for item in current_paragraphs}
                if paragraph.paragraph_id in current_ids:
                    continue
                current_paragraphs.append(paragraph)
                current_ids.add(paragraph.paragraph_id)

            chunk = self._finalize_chunk(chunks, paper_id, arxiv_id, source_pdf, section, next_chunk_index, current_paragraphs)
            if chunk is not None:
                chunks.append(chunk)
                next_chunk_index += 1

        return chunks
