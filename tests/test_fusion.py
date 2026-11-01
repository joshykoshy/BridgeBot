"""RRF math: hand-computed scores and ordering."""
import pytest

from bridgebot.ragcore.fusion import reciprocal_rank_fusion


def test_rrf_scores_match_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement_over_single_top_rank():
    # "x" is 1st in one list only; "y" is 2nd in both lists -> y wins.
    fused = reciprocal_rank_fusion([["x", "y"], ["z", "y"]], k=60)
    assert fused[0][0] == "y"


def test_rrf_empty_and_k_effect():
    assert reciprocal_rank_fusion([]) == []
    small_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    assert small_k["a"] / small_k["b"] > dict(reciprocal_rank_fusion([["a", "b"]], k=60))["a"] / dict(reciprocal_rank_fusion([["a", "b"]], k=60))["b"]
