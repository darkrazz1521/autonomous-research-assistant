"""Multi-column layout reconstruction for scientific PDFs."""

from __future__ import annotations

from collections import Counter

from autonomous_research_assistant_data.models.common import CleanParagraph, ExtractedDocument


class MultiColumnLayoutEngine:
    """Detect likely multi-column corruption and annotate paragraphs conservatively."""

    def reconstruct(self, document: ExtractedDocument, paragraphs: list[CleanParagraph], enabled: bool = True) -> tuple[list[CleanParagraph], dict[str, object]]:
        if not enabled:
            return [item.model_copy(deep=True) for item in paragraphs], {
                "multi_column_pages": 0,
                "suspected_cross_column_merges": 0,
                "column_reconstruction_enabled": False,
            }

        page_counter: Counter[int] = Counter()
        block_density = {}
        for page in document.pages:
            if len(page.block_metadata) >= 2:
                x_positions = sorted(item["bbox"][0] for item in page.block_metadata if item.get("bbox"))
                if x_positions and max(x_positions) - min(x_positions) > 120:
                    page_counter[page.page_number] += 1
            block_density[page.page_number] = len(page.block_metadata)

        reconstructed: list[CleanParagraph] = []
        suspected_merges = 0
        for paragraph in paragraphs:
            candidate = paragraph.model_copy(deep=True)
            if page_counter.get(candidate.page_number, 0):
                candidate.metadata["multi_column_page"] = True
                if len(candidate.text.split()) > 80 and candidate.text.count("  ") == 0:
                    candidate.noise_classifications = list(dict.fromkeys(candidate.noise_classifications + ["multi_column_risk"]))
                    suspected_merges += 1
            candidate.metadata["page_block_density"] = block_density.get(candidate.page_number, 0)
            reconstructed.append(candidate)

        analytics = {
            "multi_column_pages": len(page_counter),
            "suspected_cross_column_merges": suspected_merges,
            "column_reconstruction_enabled": True,
        }
        return reconstructed, analytics
