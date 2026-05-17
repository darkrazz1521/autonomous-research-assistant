"""Prompt templates for scientific RAG."""

from __future__ import annotations


TEMPLATES = {
    "qa": (
        "You are a scientific research assistant. Answer the question using only the provided evidence. "
        "If evidence is incomplete, say so explicitly. Preserve citation traceability.\n\n"
        "Question:\n{query}\n\nEvidence:\n{context}\n\nAnswer:"
    ),
    "summarization": "Summarize the evidence below with grounded claims only.\n\nTopic:\n{query}\n\nEvidence:\n{context}\n\nSummary:",
    "comparison": "Compare the concepts in the question using only the evidence below.\n\nQuestion:\n{query}\n\nEvidence:\n{context}\n\nComparison:",
    "contradiction": "Analyze whether the evidence contains agreement or contradiction.\n\nQuestion:\n{query}\n\nEvidence:\n{context}\n\nAnalysis:",
    "literature_review": "Write a concise literature review using only the evidence below.\n\nTheme:\n{query}\n\nEvidence:\n{context}\n\nReview:",
}
