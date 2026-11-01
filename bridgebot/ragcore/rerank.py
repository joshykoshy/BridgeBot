"""Cross-encoder reranking.

Concept: retrievers (BM25, bi-encoder embeddings) score query and passage *independently*,
which is fast but shallow. A cross-encoder reads the query and passage *together* through
one transformer and outputs a relevance score, which is much more accurate but costs one
forward pass per pair. So we use it only on the top 20 fused candidates and keep the top 5.

The score (sigmoid of the model's logit, 0..1) is also our confidence signal: if the best
passage scores below the abstain threshold, the answer step says "my sources don't cover
this" instead of guessing.

Trade-off: adds a few hundred ms on CPU for 20 pairs. ms-marco-MiniLM-L-6-v2 was trained on
web search queries, so absolute scores are not calibrated for this domain; the threshold is
tuned on the eval set.
"""
from __future__ import annotations

import inspect
from functools import lru_cache

import numpy as np

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, device="cpu")


def _identity_kwarg(model) -> dict:
    """Ask for raw logits; the kwarg name changed between sentence-transformers versions."""
    import torch
    params = inspect.signature(model.predict).parameters
    for name in ("activation_fn", "activation_fct"):
        if name in params:
            return {name: torch.nn.Identity()}
    return {}


def score_pairs(query: str, texts: list[str]) -> list[float]:
    """Relevance in 0..1 = sigmoid(raw logit)."""
    model = _model()
    raw = model.predict([(query, t) for t in texts], show_progress_bar=False, **_identity_kwarg(model))
    return (1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=float)))).tolist()


def rerank(query: str, candidates: list[tuple[str, str]], top_n: int = 5) -> list[tuple[str, float]]:
    """candidates = [(id, text)]. Returns [(id, score)] best first, at most top_n."""
    if not candidates:
        return []
    scores = score_pairs(query, [text for _, text in candidates])
    ranked = sorted(zip([cid for cid, _ in candidates], scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
