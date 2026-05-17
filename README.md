# Autonomous Multi-Agent Scientific Research Assistant

## Overview

This repository contains the data, processing, and retrieval foundation for an autonomous scientific research assistant. The current codebase covers:

- Phase 2: environment-aware ingestion for arXiv, SciFact, FEVER, and MS MARCO
- Phase 3: scientific PDF processing into retrieval-ready chunks
- Phase 4: embeddings, vector indexes, hybrid retrieval, reranking, evaluation, and retrieval quality optimization

The project is intentionally structured so the same workflows can run:

- locally with Conda or plain Python
- in Google Colab with optional Drive persistence
- through manifest-driven resumable scripts
- without coupling the system to a single vector DB or RAG framework

What is not implemented yet:

- end-to-end RAG answer generation
- multi-agent orchestration logic
- production retrieval serving
- distributed retrieval backends

## Current Project Analysis

At a high level, the project is in a strong Phase 4 state:

- ingestion and PDF processing are already modular and environment-aware
- chunk records already carry the metadata needed for Phase 5 RAG and multi-hop reasoning
- retrieval is abstracted cleanly across embedding, vector store, hybrid search, reranking, analytics, and evaluation
- manifests and directory contracts are stable and already support resumable local and Colab runs

The main architectural strengths are:

- layered config via `configs/base.yaml`, `configs/local.yaml`, and `configs/colab.yaml`
- deterministic fallback behavior when heavyweight retrieval dependencies are unavailable
- strong artifact separation under `datasets/`
- retrieval metadata that is already graph- and context-window-friendly

The main operational caveat right now is:

- best retrieval quality still depends on the environment successfully using real retrieval dependencies such as `sentence-transformers` and `faiss-cpu`; otherwise the system falls back gracefully, but semantic quality is lower

## Runtime Architecture

Scripts load layered configuration in this order:

1. `configs/base.yaml`
2. `configs/local.yaml` or `configs/colab.yaml`
3. one optional `--config` override

Every major workflow uses the same environment-aware entrypoints under `scripts/`.

## Pipeline Architecture

### Phase 3 PDF Processing

```text
PDF -> extracted text -> advanced normalization -> repair layer -> structural sections -> equation-aware semantic chunks -> quality scoring -> analytics + validation
```

Key capabilities:

- `pymupdf` extraction with `pdfplumber` fallback
- front matter extraction
- figure and table isolation
- reference separation
- equation-aware chunking
- heading intelligence and canonical section labels
- layout-aware cleaning and repair
- retrieval-oriented chunk scoring and analytics

### Phase 4 Retrieval

```text
chunks -> quality analysis -> filtered embedding batches -> persistent vector index -> dense + sparse retrieval -> fusion -> reranking -> context expansion -> analytics + evaluation
```

Key retrieval capabilities:

- sentence-transformer abstraction with deterministic fallback
- primary embedding target: `BAAI/bge-base-en-v1.5`
- FAISS-oriented local vector indexing with numpy fallback
- BM25 sparse retrieval
- hybrid retrieval with `rrf` and weighted fusion
- reranking abstraction for cross-encoders
- section-aware weighting
- query expansion for scientific acronyms and abbreviations
- context-window retrieval using chunk neighborhood metadata
- chunk-quality filtering and cached retrieval-quality analytics
- retrieval validation and evaluation outputs

## Repository Structure

```text
project/
|-- configs/
|-- datasets/
|   |-- embeddings/
|   |-- memory_graph/
|   |-- processed/
|   |-- raw/
|   |-- rerank_cache/
|   |-- retrieval_analytics/
|   |-- retrieval_cache/
|   |-- retrieval_evaluation/
|   |-- state/
|   `-- vector_indexes/
|-- notebooks/
|-- scripts/
`-- src/autonomous_research_assistant_data/
    |-- chunking/
    |-- core/
    |-- ingestion/
    |-- models/
    |-- parsers/
    |-- processing/
    |-- retrieval/
    |   |-- analytics/
    |   |-- api/
    |   |-- context/
    |   |-- embedding/
    |   |-- evaluation/
    |   |-- memory/
    |   |-- quality/
    |   |-- query_expansion/
    |   |-- ranking/
    |   |-- rerank/
    |   |-- search/
    |   `-- vectorstores/
    |-- storage/
    |-- utils/
    |-- validation/
    `-- validators/
```

## Installation

### Conda

```powershell
conda create -n ara-data python=3.11 -y
conda activate ara-data
pip install -r requirements.txt
pip install -e .
```

### Environment File

```powershell
conda env create -f environment.yml
conda activate autonomous-research-assistant-data
```

For highest Phase 4 quality, make sure the runtime can actually install and load:

- `sentence-transformers`
- `faiss-cpu`

## Local Workflow

### Phase 2

```powershell
python scripts/bootstrap_project.py --env local
python scripts/ingest_arxiv_simple.py --env local
python scripts/ingest_datasets.py --env local --datasets scifact fever msmarco
python scripts/validate_phase2.py --env local
```

Alternative arXiv ingestion remains available through:

- `scripts/ingest_arxiv.py`
- `scripts/ingest_arxiv_simple.py`

### Phase 3

```powershell
python scripts/process_arxiv_pdfs.py --env local
python scripts/validate_pdf_processing.py --env local
python scripts/review_chunks.py --env local --sample-size 8
```

Optional config override:

```powershell
python scripts/process_arxiv_pdfs.py --env local --config configs/my_processing_override.yaml
```

### Phase 4

Generate embeddings:

```powershell
python scripts/generate_embeddings.py --env local
python scripts/generate_embeddings.py --env local --quality-filtering
```

Build the vector index:

```powershell
python scripts/build_vector_index.py --env local
```

Run retrieval:

```powershell
python scripts/run_retrieval_query.py --env local --query "What is GRPO?" --hybrid --rerank
python scripts/run_retrieval_query.py --env local --query "What is GRPO?" --hybrid --rerank --expand-query --context-window
```

Evaluate retrieval:

```powershell
python scripts/evaluate_retrieval.py --env local --hybrid --rerank
```

Validate retrieval quality:

```powershell
python scripts/validate_retrieval_quality.py --env local
```

## Retrieval CLI Extensions

The retrieval scripts now support the following quality-optimization flags:

- `--quality-filtering`
- `--disable-section-weighting`
- `--context-window`
- `--window-radius`
- `--fusion-method`
- `--expand-query`
- `--strict-retrieval-validation`

Core existing flags remain unchanged:

- `--force-rebuild`
- `--embedding-model`
- `--vector-db`
- `--batch-size`
- `--top-k`
- `--hybrid`
- `--rerank`
- `--section-filter`

## Retrieval Quality Optimization Layer

Phase 4 now includes an explicit retrieval quality optimization layer with these additions:

### Chunk Quality Analysis

- numeric ratio
- alphabetic ratio
- semantic density
- language entropy
- table probability
- benchmark probability
- duplicate line ratio
- citation density
- equation density
- average sentence length
- malformed structure scoring
- final retrieval quality and noise scores

Quality analytics are cached under:

- `datasets/retrieval_analytics/chunk_quality/`
- `datasets/retrieval_analytics/chunk_quality_summary.json`

### Section-Aware Weighting

Section weighting is configurable and currently prioritizes:

- abstract
- introduction
- methodology and methods
- results
- discussion
- conclusion

It downweights:

- appendix
- references

### Context-Window Retrieval

Retrieved results can now include:

- `primary_chunk`
- `context_before`
- `context_after`
- `merged_context`

These are built from:

- `previous_chunk_id`
- `next_chunk_id`
- `parent_section_id`

### Query Expansion

The query expansion layer provides local heuristic expansion for scientific terms such as:

- `GRPO`
- `RAG`
- `LLM`
- `RLHF`

Results include a `query_expansion_report` with:

- `original_query`
- `expanded_terms`
- `rewritten_query`

### Hybrid Fusion and Score Breakdown

Retrieval scoring now exposes:

- `raw_vector_score`
- `raw_sparse_score`
- `rerank_score`
- `section_weight`
- `final_retrieval_score`
- `final_score_breakdown`

## Colab Support

The Colab workflow supports:

- runtime detection
- optional Google Drive mounting
- `/content` and Drive-aware storage
- Colab-specific config in `configs/colab.yaml`
- notebook-driven orchestration with script-based reproducibility

Start with:

- [notebooks/README.md](/C:/Users/siddh/ML_projects/research/notebooks/README.md)
- [notebooks/setup_colab.ipynb](/C:/Users/siddh/ML_projects/research/notebooks/setup_colab.ipynb)

## Output Artifacts

### Phase 3 Outputs

- front matter JSON under `datasets/processed/front_matter/`
- extracted document JSON under `datasets/processed/extracted_text/`
- cleaned text JSON under `datasets/processed/cleaned_text/`
- repaired text JSON under `datasets/processed/repaired_text/`
- section JSON under `datasets/processed/sections/`
- chunk JSON under `datasets/processed/chunks/`
- citation JSON under `datasets/processed/citations/`
- equation block JSON under `datasets/processed/equation_blocks/`
- heading analysis JSON under `datasets/processed/heading_analysis/`
- isolated figures and tables under `datasets/processed/isolated_figures/` and `datasets/processed/isolated_tables/`
- references under `datasets/processed/references/`
- validation reports under `datasets/processed/validation/`
- processing summaries under `datasets/processed/reports/`
- processing analytics under `datasets/processed/analytics/`

### Phase 4 Outputs

- embedding artifacts under `datasets/embeddings/`
- vector indexes under `datasets/vector_indexes/`
- rerank caches under `datasets/rerank_cache/`
- retrieval traces under `datasets/retrieval_analytics/query_traces.jsonl`
- retrieval analytics under `datasets/retrieval_analytics/`
- retrieval evaluation reports under `datasets/retrieval_evaluation/`
- retrieval quality reports under `datasets/retrieval_evaluation/quality_reports/`
- memory graph outputs under `datasets/memory_graph/`

## Operational Notes

- ingestion state is tracked in `datasets/state/ingestion_state.json`
- PDF processing state is tracked in `datasets/state/processing_state.json`
- retrieval manifests remain under `datasets/vector_indexes/manifests/`
- processing and retrieval are designed to be resumable
- the current vector-store abstraction is FAISS-first but leaves room for Qdrant, LanceDB, and Chroma
- the retrieval layer is now structurally ready for Phase 5 citation-safe RAG and multi-agent reasoning
