"""Paragraph deduplication for scientific corpora."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from autonomous_research_assistant_data.models.common import CleanParagraph, DedupGroup


class ParagraphDedupEngine:
    """Deduplicate repeated paragraph fragments with adjacency-aware similarity."""

    def _fingerprint(self, text: str) -> str:
        normalized = re.sub(r"\W+", " ", text.lower()).strip()
        shingles = [normalized[idx : idx + 48] for idx in range(0, max(len(normalized) - 47, 1), 16)]
        payload = "|".join(shingles[:12]) or normalized
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _similarity(self, left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[a-z0-9]{3,}", left.lower()))
        right_tokens = set(re.findall(r"[a-z0-9]{3,}", right.lower()))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)

    def deduplicate(self, paper_id: str, arxiv_id: str, paragraphs: list[CleanParagraph], strict: bool = False) -> tuple[list[CleanParagraph], list[DedupGroup], dict[str, int]]:
        buckets: dict[str, list[CleanParagraph]] = defaultdict(list)
        for paragraph in paragraphs:
            buckets[self._fingerprint(paragraph.text)].append(paragraph)

        groups: list[DedupGroup] = []
        groups_by_key: dict[tuple[str, ...], str] = {}
        cleaned: list[CleanParagraph] = []
        duplicate_count = 0

        for paragraph in paragraphs:
            candidate = paragraph.model_copy(deep=True)
            matches = buckets[self._fingerprint(candidate.text)]
            if len(matches) > 1:
                key = tuple(sorted(item.paragraph_id for item in matches))
                group_id = groups_by_key.get(key)
                if group_id is None:
                    group_id = f"{paper_id}-dup{len(groups):04d}"
                    groups_by_key[key] = group_id
                    groups.append(
                        DedupGroup(
                            group_id=group_id,
                            paper_id=paper_id,
                            arxiv_id=arxiv_id,
                            paragraph_ids=[item.paragraph_id for item in matches],
                            canonical_paragraph_id=matches[0].paragraph_id,
                            dedup_confidence=0.98,
                            overlap_cluster=f"page-{candidate.page_number}",
                            metadata={"exact_hash_group": True},
                        )
                    )
                candidate.duplicate_group_id = group_id
                candidate.dedup_confidence = 0.98
                candidate.overlap_cluster = f"page-{candidate.page_number}"
                if candidate.paragraph_id != matches[0].paragraph_id:
                    duplicate_count += 1
                    if strict:
                        continue
            else:
                for prior in cleaned[-3:]:
                    similarity = self._similarity(candidate.text, prior.text)
                    if similarity >= (0.9 if strict else 0.96):
                        candidate.duplicate_group_id = prior.duplicate_group_id or f"{paper_id}-dup{len(groups):04d}"
                        candidate.dedup_confidence = round(similarity, 4)
                        candidate.overlap_cluster = f"adjacent-{candidate.page_number}"
                        duplicate_count += 1
                        if strict:
                            break
                if strict and candidate.duplicate_group_id:
                    continue
            cleaned.append(candidate)

        return cleaned, groups, {
            "duplicate_paragraph_count": duplicate_count,
            "duplicate_group_count": len(groups),
        }
