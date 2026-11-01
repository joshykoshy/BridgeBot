"""Retrieval metrics at the answer level."""
import math

from bridgebot.ragcore.evals import mrr, ndcg_at_k, recall_at_k


def test_metrics():
    retrieved = ["x", "a#2", "y", "a#3"]
    relevant = {"a#2", "a#3"}  # two overlapping chunks holding the same answer
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert mrr(retrieved, relevant) == 0.5
    assert math.isclose(ndcg_at_k(retrieved, relevant), 1 / math.log2(3))
    assert mrr(["x"], relevant) == 0.0 and ndcg_at_k(["x"], relevant) == 0.0
