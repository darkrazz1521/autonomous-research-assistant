from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomous_research_assistant_data.cli import build_common_parser, load_config_from_args
from autonomous_research_assistant_data.storage.file_store import read_json


def _load_chunks(chunks_dir: Path) -> list[dict]:
    chunks: list[dict] = []
    for chunk_file in chunks_dir.rglob("*.json"):
        payload = read_json(chunk_file)
        if isinstance(payload.get("chunks"), list):
            for chunk in payload["chunks"]:
                chunk["_file"] = str(chunk_file)
                chunks.append(chunk)
    return chunks


def _preview(chunk: dict, previous_chunk: dict | None = None) -> dict:
    overlap = []
    if previous_chunk:
        overlap = [pid for pid in chunk.get("paragraph_ids", []) if pid in set(previous_chunk.get("paragraph_ids", []))]
    return {
        "chunk_id": chunk.get("chunk_id"),
        "section_name": chunk.get("section_name"),
        "retrieval_quality_score": chunk.get("retrieval_quality_score"),
        "repair_confidence": chunk.get("repair_confidence"),
        "structural_integrity_score": chunk.get("structural_integrity_score"),
        "transition_quality_score": chunk.get("transition_quality_score"),
        "semantic_boundary_score": chunk.get("semantic_boundary_score"),
        "narrative_continuity_score": chunk.get("narrative_continuity_score"),
        "contains_equation": chunk.get("contains_equation"),
        "contains_citation": chunk.get("contains_citation"),
        "citation_density": chunk.get("citation_density"),
        "equation_density": chunk.get("equation_density"),
        "noise_classifications": chunk.get("noise_classifications", []),
        "corruption_categories": chunk.get("corruption_categories", []),
        "repair_recommendations": chunk.get("repair_recommendations", []),
        "paragraph_overlap_count": len(overlap),
        "paragraph_ids": chunk.get("paragraph_ids", [])[:8],
        "topic_signature": chunk.get("chunk_topic_signature", []),
        "preview": chunk.get("chunk_text", "")[:900],
        "file": chunk.get("_file"),
    }


def main() -> None:
    parser = build_common_parser("Review processed chunks with random sampling and overlap inspection.")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of chunks to sample.")
    parser.add_argument("--paper-id", help="Optional paper id filter.")
    parser.add_argument("--min-quality", type=float, default=0.0, help="Minimum retrieval quality score.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    chunks = _load_chunks(config.pdf_processing.chunks_dir)
    if args.paper_id:
        chunks = [chunk for chunk in chunks if chunk.get("paper_id") == args.paper_id]
    chunks = [chunk for chunk in chunks if float(chunk.get("retrieval_quality_score", 0.0)) >= args.min_quality]
    chunks.sort(key=lambda item: (item.get("paper_id", ""), item.get("chunk_index", 0)))

    if not chunks:
        print(json.dumps({"message": "No chunks matched the review criteria."}, indent=2))
        return

    sample = random.sample(chunks, min(args.sample_size, len(chunks)))
    sample.sort(key=lambda item: (item.get("paper_id", ""), item.get("chunk_index", 0)))

    previous_by_paper: dict[str, dict] = {}
    previews = []
    for chunk in sample:
        paper_id = chunk.get("paper_id", "")
        previews.append(_preview(chunk, previous_by_paper.get(paper_id)))
        previous_by_paper[paper_id] = chunk

    section_integrity = {}
    for chunk in chunks:
        section_integrity.setdefault(chunk.get("section_name", "unknown"), 0)
        section_integrity[chunk.get("section_name", "unknown")] += 1

    print(
        json.dumps(
            {
                "total_chunks_considered": len(chunks),
                "sample_size": len(previews),
                "section_integrity": section_integrity,
                "samples": previews,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
