# Autonomous Multi-Agent Scientific Research Assistant

## Phases 2-4 Infrastructure

This repository provides a production-grade scientific data and retrieval platform with:

- Phase 2: environment-aware ingestion for arXiv, SciFact, FEVER, and MS MARCO
- Phase 3: elite scientific PDF processing for retrieval-ready corpus generation
- Phase 4: embedding, vector retrieval, hybrid search, reranking, and retrieval analytics infrastructure

The system is designed for:

- local development with VS Code and Conda
- version control with GitHub
- heavier execution in Google Colab
- future compatibility with scientific RAG, reranking, hybrid retrieval, citation grounding, and agent memory systems

It intentionally does not yet implement:

- full RAG orchestration
- agent frameworks
- graph databases
- distributed vector search infrastructure
- production serving for retrieval APIs

## Runtime Architecture

The package uses layered environment-aware configuration:

- `configs/base.yaml`
- `configs/local.yaml`
- `configs/colab.yaml`

Scripts load `base.yaml` plus the selected environment config, and may optionally apply one additional override file with `--config`.

## Elite PDF Processing Architecture

Phase 3 implements a high-quality scientific corpus pipeline:

`PDF -> extracted text -> advanced normalization -> repair layer -> structural sections -> equation-aware semantic chunks -> quality scoring -> analytics + validation`

Key PDF capabilities:

- `pymupdf` extraction with `pdfplumber` fallback
- front matter extraction for title, authors, affiliations, abstract, emails, links, and keywords
- figure/table caption filtering and excluded-artifact tracking
- reference section isolation into separate artifacts
- equation-aware paragraph preservation
- structural heading intelligence and canonical section labeling
- layout-aware figure/table isolation and multi-column reconstruction
- adaptive chunk sizing with minimized overlap duplication
- chunk quality scoring for coherence, noise, structure, semantic density, and retrieval readiness
- resumable manifest-based processing for Local and Colab workflows

## Retrieval Architecture

Phase 4 adds a retrieval-grade scientific memory layer:

`chunks -> filtered embedding batches -> persistent vector index -> dense + sparse retrieval -> hybrid fusion -> reranking -> retrieval analytics + evaluation`

Key retrieval capabilities:

- sentence-transformers model abstraction with deterministic fallback
- primary embedding target: `BAAI/bge-base-en-v1.5`
- persistent FAISS-ready local vector store with numpy fallback
- manifest-driven embedding and index rebuilds
- hybrid dense + BM25 retrieval
- citation-aware and section-aware scoring
- reranking abstraction for cross-encoders
- graph-ready neighbor metadata for future memory systems
- retrieval evaluation and analytics artifacts

## Folder Structure

```text
project/
├── configs/
├── datasets/
│   ├── raw/
│   │   └── arxiv/
│   │       ├── metadata/
│   │       └── pdfs/
│   ├── processed/
│   │   ├── analytics/
│   │   ├── chunks/
│   │   ├── citations/
│   │   ├── cleaned_text/
│   │   ├── dedup_reports/
│   │   ├── equation_blocks/
│   │   ├── extracted_text/
│   │   ├── front_matter/
│   │   ├── heading_analysis/
│   │   ├── isolated_figures/
│   │   ├── isolated_tables/
│   │   ├── manifests/
│   │   ├── references/
│   │   ├── repaired_text/
│   │   ├── repair_reports/
│   │   ├── reports/
│   │   ├── sections/
│   │   └── validation/
│   ├── embeddings/
│   ├── memory_graph/
│   ├── rerank_cache/
│   ├── retrieval_analytics/
│   ├── retrieval_cache/
│   ├── retrieval_evaluation/
│   ├── state/
│   └── vector_indexes/
├── notebooks/
├── scripts/
│   ├── bootstrap_project.py
│   ├── build_vector_index.py
│   ├── evaluate_retrieval.py
│   ├── generate_embeddings.py
│   ├── ingest_arxiv.py
│   ├── ingest_arxiv_simple.py
│   ├── ingest_datasets.py
│   ├── process_arxiv_pdfs.py
│   ├── review_chunks.py
│   ├── run_retrieval_query.py
│   ├── validate_pdf_processing.py
│   └── validate_phase2.py
└── src/autonomous_research_assistant_data/
    ├── chunking/
    ├── core/
    ├── ingestion/
    ├── models/
    ├── parsers/
    ├── processing/
    ├── retrieval/
    ├── storage/
    ├── utils/
    ├── validators/
    └── validation/
```

## Install

```powershell
conda create -n ara-data python=3.11 -y
conda activate ara-data
pip install -r requirements.txt
pip install -e .
```

Or:

```powershell
conda env create -f environment.yml
conda activate autonomous-research-assistant-data
```

For full Phase 4 retrieval quality, ensure the environment can install the retrieval dependencies in `requirements.txt`, especially `sentence-transformers` and `faiss-cpu`.

## Phase 2 Usage

```powershell
python scripts/bootstrap_project.py --env local
python scripts/ingest_arxiv_simple.py --env local
python scripts/ingest_datasets.py --env local --datasets scifact fever msmarco
python scripts/validate_phase2.py --env local
```

Two arXiv ingestion modes remain available:

- `scripts/ingest_arxiv.py`
- `scripts/ingest_arxiv_simple.py`

## Phase 3 Usage

Process PDFs:

```powershell
python scripts/process_arxiv_pdfs.py --env local
```

Validate outputs:

```powershell
python scripts/validate_pdf_processing.py --env local
```

Review chunks:

```powershell
python scripts/review_chunks.py --env local --sample-size 8
```

Optional override:

```powershell
python scripts/process_arxiv_pdfs.py --env local --config configs/my_processing_override.yaml
```

## Phase 4 Usage

Generate embeddings:

```powershell
python scripts/generate_embeddings.py --env local
```

Build the vector index:

```powershell
python scripts/build_vector_index.py --env local
```

Run a retrieval query:

```powershell
python scripts/run_retrieval_query.py --env local --query "What is GRPO?" --hybrid --rerank
```

Evaluate retrieval:

```powershell
python scripts/evaluate_retrieval.py --env local
```

Optional retrieval flags:

- `--force-rebuild`
- `--embedding-model`
- `--vector-db`
- `--batch-size`
- `--top-k`
- `--hybrid`
- `--rerank`
- `--section-filter`

## Colab Support

The package includes:

- runtime detection
- optional Google Drive mounting
- `/content/` and `/content/drive/MyDrive/` path support
- GPU availability inspection
- batch-oriented processing and retrieval limits

See [setup_colab.ipynb](/C:/Users/siddh/ML_projects/research/notebooks/setup_colab.ipynb).

## Output Artifacts

Phase 3 writes:

- front matter JSON under `datasets/processed/front_matter/`
- extracted document JSON under `datasets/processed/extracted_text/`
- cleaned text and excluded-artifact JSON under `datasets/processed/cleaned_text/`
- repaired text JSON under `datasets/processed/repaired_text/`
- section JSON under `datasets/processed/sections/`
- chunk JSON under `datasets/processed/chunks/`
- citation JSON under `datasets/processed/citations/`
- equation block JSON under `datasets/processed/equation_blocks/`
- heading analysis JSON under `datasets/processed/heading_analysis/`
- isolated figure/table region JSON under `datasets/processed/isolated_figures/` and `datasets/processed/isolated_tables/`
- dedup and repair reports under `datasets/processed/dedup_reports/` and `datasets/processed/repair_reports/`
- isolated references under `datasets/processed/references/`
- validation reports under `datasets/processed/validation/`
- processing summaries under `datasets/processed/reports/`
- corpus analytics under `datasets/processed/analytics/`

Phase 4 writes:

- embedding artifacts under `datasets/embeddings/`
- persistent vector indexes under `datasets/vector_indexes/`
- retrieval query caches under `datasets/retrieval_cache/`
- rerank caches under `datasets/rerank_cache/`
- retrieval analytics under `datasets/retrieval_analytics/`
- graph-ready memory metadata under `datasets/memory_graph/`
- retrieval evaluation reports under `datasets/retrieval_evaluation/`

Each chunk now includes future-retrieval metadata such as:

- `semantic_hash`
- `parent_section_id`
- `previous_chunk_id`
- `next_chunk_id`
- `chunk_topic_signature`
- `contains_equation`
- `contains_citation`
- `coherence_score`
- `noise_score`
- `structure_score`
- `semantic_density_score`
- `retrieval_quality_score`
- `citation_density`
- `equation_density`
- `repair_confidence`
- `structural_integrity_score`
- `transition_quality_score`
- `semantic_boundary_score`
- `narrative_continuity_score`
- `canonical_section_label`

## Review and Analytics

The processing stack produces:

- chunk-level retrieval-quality heuristics
- token histograms
- section distribution summaries
- quality histograms
- low-quality chunk flags
- overlap-aware review previews through `scripts/review_chunks.py`

The retrieval stack produces:

- embedding coverage reports
- vector density and duplicate-vector summaries
- query latency traces
- hybrid score distribution reports
- retrieval evaluation metrics such as recall@k, MRR, and nDCG
- graph-ready neighbor metadata for future memory systems

## Operational Notes

- ingestion state remains in `datasets/state/ingestion_state.json`
- PDF processing state remains in `datasets/state/processing_state.json`
- retrieval manifests remain under `datasets/vector_indexes/manifests/`
- duplicate detection is manifest-based across ingestion, processing, embeddings, and indexing
- processed artifacts are designed to feed embedding, reranking, citation-grounding, and multi-hop retrieval systems without binding the project to a single vector database or RAG framework
- the retrieval layer currently supports FAISS-oriented local indexing with an abstraction surface prepared for Qdrant, LanceDB, and Chroma
