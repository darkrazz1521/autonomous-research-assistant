# Autonomous Multi-Agent Scientific Research Assistant

## Phase 2: Data Collection Infrastructure

This repository contains a production-oriented data ingestion and organization layer for the research assistant platform. Phase 2 focuses only on:

- research paper ingestion from arXiv
- benchmark dataset ingestion for SciFact, FEVER, and MS MARCO
- metadata management
- validation, resumability, logging, and storage organization

It intentionally does not include:

- frontend applications
- retrieval pipelines
- vector databases
- embeddings
- evaluation agents
- hallucination detection models

## Architecture

The codebase uses a clean, service-oriented layout that is compatible with future FastAPI integration:

- `configs/`: central YAML configuration profiles
- `datasets/`: raw, processed, metadata, external, and state storage
- `logs/`: structured JSONL logs and failure traces
- `scripts/`: operational entry points
- `src/autonomous_research_assistant_data/`: reusable application package

Core design goals:

- modular, config-driven ingestion
- async-ready I/O for arXiv downloads
- resumable workflows with persisted state
- duplicate detection through manifests
- cloud- and Kaggle-friendly path strategy
- future compatibility with RAG and agentic workflows

## Folder Structure

```text
project/
├── configs/
│   ├── base.yaml
│   └── kaggle.yaml
├── datasets/
│   ├── external/
│   ├── metadata/
│   │   ├── manifests/
│   │   └── schemas/
│   ├── processed/
│   ├── raw/
│   │   ├── arxiv/
│   │   │   ├── metadata/
│   │   │   └── pdfs/
│   │   ├── fever/
│   │   ├── msmarco/
│   │   └── scifact/
│   └── state/
├── logs/
│   ├── failed/
│   └── ingestion/
├── notebooks/
├── scripts/
│   ├── bootstrap_project.py
│   ├── ingest_arxiv.py
│   ├── ingest_datasets.py
│   └── validate_phase2.py
├── src/
│   └── autonomous_research_assistant_data/
│       ├── config.py
│       ├── core/
│       ├── ingestion/
│       ├── models/
│       ├── storage/
│       ├── utils/
│       └── validation/
└── requirements.txt
```

## Setup

1. Create an environment.
2. Install dependencies from `requirements.txt`.
3. Review `configs/base.yaml`.
4. Run the bootstrap script to create directories.
5. Launch source-specific ingestion scripts.

Example:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/bootstrap_project.py --config configs/base.yaml
python scripts/ingest_arxiv.py --config configs/base.yaml
python scripts/ingest_datasets.py --config configs/base.yaml --datasets scifact fever msmarco
python scripts/validate_phase2.py --config configs/base.yaml
```

## Kaggle Workflow

For low local storage usage:

- keep only code locally
- override paths with `configs/kaggle.yaml` when running in Kaggle
- rely on Hugging Face cache and exported parquet artifacts in mounted Kaggle storage
- use incremental arXiv syncs instead of full historical downloads

Example in Kaggle:

```python
!python scripts/bootstrap_project.py --config configs/kaggle.yaml
!python scripts/ingest_arxiv.py --config configs/kaggle.yaml
!python scripts/ingest_datasets.py --config configs/kaggle.yaml --datasets scifact fever msmarco
```

## Operational Notes

- arXiv state is persisted in `datasets/state/ingestion_state.json`
- duplicate detection is managed through manifests in `datasets/metadata/manifests/`
- metadata is written per record as JSON and summarized in manifests
- all ingestion runs emit structured JSONL logs under `logs/`

