## Notebook Guidance

Notebooks in this repository should stay orchestration-focused. Heavy downloads, ingestion, PDF processing, embedding generation, and retrieval evaluation should still run through `scripts/` so runs remain reproducible across local development and Google Colab.

## Current Notebook Role

The Colab notebook is now intended to help with:

- cloning the repository into `/content`
- installing project dependencies
- mounting Google Drive when persistence is needed
- running the main Phase 2 to Phase 4 scripts in a documented order
- validating that Colab configuration and storage paths are working

## Recommended Colab Flow

1. Open [setup_colab.ipynb](/C:/Users/siddh/ML_projects/research/notebooks/setup_colab.ipynb).
2. Set `REPO_URL` and `REPO_DIR`.
3. Install requirements and the editable package.
4. Mount Google Drive if you want persistence across sessions.
5. Run bootstrap and ingestion.
6. Process PDFs.
7. Generate embeddings and build the vector index.
8. Run a sample retrieval query with query expansion and context windows.
9. Run retrieval evaluation and retrieval quality validation.

## Colab Notes

- Use `--env colab` for all notebook-run scripts.
- `configs/colab.yaml` already lowers batch sizes and adjusts paths for `/content`.
- If `sentence-transformers` or `faiss-cpu` are unavailable in the Colab runtime, retrieval still runs with graceful fallbacks, but quality will be lower.
