from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autonomous_research_assistant_data.bootstrap import bootstrap_directories, prepare_runtime
from autonomous_research_assistant_data.cli import build_common_parser, load_config_from_args
from autonomous_research_assistant_data.core.logging import configure_logging
from autonomous_research_assistant_data.core.time import utc_now
from autonomous_research_assistant_data.retrieval.common import load_chunk_records, load_embedding_records
from autonomous_research_assistant_data.retrieval.quality.chunk_quality import ChunkQualityAnalyzer
from autonomous_research_assistant_data.storage.file_store import ensure_directory, write_json


def main() -> None:
    parser = build_common_parser("Validate retrieval quality artifacts and detect noisy retrieval risks.")
    parser.add_argument("--embedding-model", help="Override the configured embedding model.")
    args = parser.parse_args()

    config = load_config_from_args(args)
    configure_logging(config)
    prepare_runtime(config)
    bootstrap_directories(config)

    chunks = [chunk for _, chunk in load_chunk_records(config.pdf_processing.chunks_dir)]
    chunk_records = load_chunk_records(config.pdf_processing.chunks_dir)
    embeddings = load_embedding_records(config.retrieval.embeddings_dir, args.embedding_model or config.retrieval.embedding.default_model)
    analyzer = ChunkQualityAnalyzer(config)
    embedding_ids = [record.chunk_id for record in embeddings]
    duplicate_embeddings = [chunk_id for chunk_id, count in Counter(embedding_ids).items() if count > 1]

    malformed_contexts: list[str] = []
    ordering_issues: list[str] = []
    repeated_chunks: list[str] = []
    citation_corruption: list[str] = []
    table_heavy: list[str] = []
    low_density: list[str] = []
    overlap_explosion: list[str] = []
    seen_hashes: dict[str, str] = {}
    by_paper: dict[str, list] = defaultdict(list)

    for path, chunk in chunk_records:
        quality = analyzer.analyse(chunk, source_path=path)
        by_paper[chunk.paper_id].append(chunk)
        if float(quality.get("table_probability", 0.0)) > config.retrieval.quality.max_table_probability:
            table_heavy.append(chunk.chunk_id)
        if float(quality.get("semantic_density_score", 0.0)) < config.retrieval.quality.min_semantic_density:
            low_density.append(chunk.chunk_id)
        if chunk.contains_citation and not chunk.citation_spans:
            citation_corruption.append(chunk.chunk_id)
        if chunk.previous_chunk_id == chunk.chunk_id or chunk.next_chunk_id == chunk.chunk_id:
            malformed_contexts.append(chunk.chunk_id)
        if chunk.semantic_hash and chunk.semantic_hash in seen_hashes:
            repeated_chunks.append(chunk.chunk_id)
        elif chunk.semantic_hash:
            seen_hashes[chunk.semantic_hash] = chunk.chunk_id
        if len(set(chunk.paragraph_ids)) != len(chunk.paragraph_ids):
            overlap_explosion.append(chunk.chunk_id)

    for paper_id, paper_chunks in by_paper.items():
        paper_chunks.sort(key=lambda item: item.chunk_index)
        for index, chunk in enumerate(paper_chunks[:-1]):
            expected = paper_chunks[index + 1].chunk_id
            if chunk.next_chunk_id and chunk.next_chunk_id != expected:
                ordering_issues.append(chunk.chunk_id)

    report = {
        "generated_at": utc_now(),
        "total_chunks": len(chunks),
        "embedded_records": len(embeddings),
        "duplicate_retrievals": duplicate_embeddings,
        "excessive_table_retrievals": table_heavy,
        "low_semantic_density_chunks": low_density,
        "repeated_chunks": repeated_chunks,
        "citation_corruption": citation_corruption,
        "malformed_contexts": malformed_contexts,
        "context_ordering_issues": ordering_issues,
        "chunk_overlap_explosion": overlap_explosion,
        "summary": {
            "duplicate_retrieval_count": len(duplicate_embeddings),
            "table_heavy_count": len(table_heavy),
            "low_density_count": len(low_density),
            "repeated_chunk_count": len(repeated_chunks),
            "citation_corruption_count": len(citation_corruption),
            "malformed_context_count": len(malformed_contexts),
            "ordering_issue_count": len(ordering_issues),
            "overlap_explosion_count": len(overlap_explosion),
        },
    }
    output_dir = ensure_directory(config.retrieval.retrieval_evaluation_dir / "quality_reports")
    output_path = output_dir / f"retrieval_quality_validation_{utc_now().strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(output_path, report)
    print(json.dumps({"report_path": str(output_path), **report["summary"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
