"""Reciprocal Rank Fusion (RRF): merges several ranked lists into one.

Why: BM25 scores (unbounded term statistics) and cosine similarities (0..1) live on
different scales, so adding them is meaningless. RRF ignores the scores and uses only
ranks: score(d) = sum over lists of 1 / (k + rank_d). A document that is near the top
of *any* list rises; a document near the top of *several* lists rises most.

Trade-off: RRF throws away score magnitude (a 0.99 match and a 0.51 match at rank 1 count
the same). k=60 is the value from the original paper (Cormack et al., 2009); a larger k
flattens the curve so rank 1 vs rank 5 matters less, a smaller k rewards the very top.
"""
from __future__ import annotations


def reciprocal_rank_fusion(lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids. Returns (id, rrf_score) sorted best first."""
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
