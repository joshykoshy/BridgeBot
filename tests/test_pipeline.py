"""Retrieval pipeline behaviours named in V1_TO_V2.md. 'slow' tests load the real index/models."""
import os

import pytest

from bridgebot.ragcore.rewrite import offline_rewrite

os.environ.setdefault("OFFLINE", "1")


def test_offline_rewrite_resolves_follow_up():
    history = [{"role": "user", "content": "I'm visiting a family in Japan."}, {"role": "assistant", "content": "Nice."}]
    rw = offline_rewrite("Which gifts should I avoid there?", history, vocab=["Japan", "China"])
    assert "Japan" in rw.standalone


@pytest.fixture(scope="module")
def indexes():
    from bridgebot.corpus import get_indexes
    return get_indexes()


@pytest.mark.slow
def test_index_reused_when_corpus_unchanged(indexes):
    from bridgebot.config import INDEX_DIR
    from bridgebot.corpus import load_chunks
    from bridgebot.ragcore.pipeline import Indexes
    assert Indexes(load_chunks(), INDEX_DIR, collection="bridgebot").rebuilt is False


@pytest.mark.slow
def test_hybrid_finds_exact_name(indexes):
    from bridgebot.ragcore.pipeline import RagPipeline
    res = RagPipeline(indexes, use_rewrite=False).retrieve("What is otoshidama?")
    assert any(p.id.startswith("red-envelope#") for p in res.passages[:3])


@pytest.mark.slow
def test_abstains_off_topic(indexes):
    from bridgebot.config import ABSTAIN_THRESHOLD
    from bridgebot.ragcore.pipeline import RagPipeline
    pipe = RagPipeline(indexes, use_rewrite=False, abstain_threshold=ABSTAIN_THRESHOLD)
    assert pipe.retrieve("How do I reset my wifi router password?").abstain
    assert not pipe.retrieve("How should I greet my host in Japan?").abstain
