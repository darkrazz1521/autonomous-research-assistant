# Autonomous Multi-Agent Scientific Research Assistant

## Phase 2: Data Collection Infrastructure

This repository contains the production-grade data ingestion and organization layer for the research assistant platform. Phase 2 focuses only on:

- research paper ingestion from arXiv
- benchmark dataset ingestion for SciFact, FEVER, and MS MARCO
- metadata management
- validation, resumability, logging, and structured storage

It intentionally does not include:

- frontend applications
- retrieval pipelines
- vector databases
- embeddings
- evaluation agents
- hallucination detection models

## Runtime Architecture

The codebase now targets a Local + Google Colab AI engineering workflow:

- local development with VS Code and Conda
- version control with GitHub
- heavy processing in Google Colab
- model and artifact interoperability with Hugging Face Hub

The package uses layered environment-aware configuration:

- `configs/base.yaml`: shared defaults
- `configs/local.yaml`: local development overrides
- `configs/colab.yaml`: Google Colab overrides

At runtime, scripts load `base.yaml` plus the selected environment config, and optionally apply one extra override file with `--config`.

## Folder Structure

```text
project/
├── configs/
│   ├── base.yaml
│   ├── colab.yaml
│   └── local.yaml
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
│   ├── README.md
│   └── setup_colab.ipynb
├── scripts/
│   ├── bootstrap_project.py
│   ├── ingest_arxiv.py
│   ├── ingest_arxiv_simple.py
│   ├── ingest_datasets.py
│   └── validate_phase2.py
├── src/
│   └── autonomous_research_assistant_data/
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

## Install

Local editable install:

```powershell
conda create -n ara-data python=3.11 -y
conda activate ara-data
pip install -r requirements.txt
pip install -e .
```

Conda environment:

```powershell
conda env create -f environment.yml
conda activate autonomous-research-assistant-data
```

## Environment-Aware Usage

Local workflow:

```powershell
python scripts/bootstrap_project.py --env local
python scripts/ingest_arxiv_simple.py --env local
python scripts/ingest_datasets.py --env local --datasets scifact fever msmarco
python scripts/validate_phase2.py --env local
```

Google Colab workflow:

```python
!python scripts/bootstrap_project.py --env colab
!python scripts/ingest_arxiv_simple.py --env colab
!python scripts/ingest_datasets.py --env colab --datasets scifact fever msmarco
!python scripts/validate_phase2.py --env colab
```

Optional layered override:

```powershell
python scripts/ingest_datasets.py --env local --config configs/my_experiment.yaml
```

## Colab Support

The package includes Colab utilities for:

- runtime detection
- optional Google Drive mounting
- `/content/` and `/content/drive/MyDrive/` storage paths
- GPU availability inspection
- environment-specific bootstrap behavior

See [setup_colab.ipynb](/C:/Users/siddh/ML_projects/research/notebooks/setup_colab.ipynb) for the guided Colab workflow.

## Ingestion Modes

Two arXiv ingestion paths are preserved:

- `scripts/ingest_arxiv.py`: advanced async ingestion pipeline
- `scripts/ingest_arxiv_simple.py`: conservative sequential fallback using the official `arxiv` package

The dataset ingestion system is now more resilient for modern Hugging Face dataset loading:

- layered compatibility checks
- `datasets==2.19.1` target support
- explicit logging for `trust_remote_code` compatibility
- fallback strategies for legacy dataset script cases
- exported metadata and manifest tracking for every dataset split

## Operational Notes

- arXiv and dataset state is persisted in `datasets/state/ingestion_state.json`
- duplicate detection is managed through manifests in `datasets/metadata/manifests/`
- metadata is written per record as JSON and summarized in parquet manifests
- structured ingestion and failure logs are written under `logs/`
- the storage layout stays compatible with future RAG, vector DB, and agentic AI phases
