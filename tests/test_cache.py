"""Semantic cache keying: similar query hits; different entities or corpus never hit."""
import numpy as np

from bridgebot.ragcore.cache import SemanticCache, normalise

VOCAB = ["greet", "greeting", "japan", "china", "people", "how", "do", "i", "in", "etiquette", "weather"]


def fake_embed(texts):
    """Bag-of-words over a tiny vocab; deterministic and good enough to test keying."""
    out = []
    for t in texts:
        v = np.array([t.split().count(w) for w in VOCAB], dtype=np.float32) + 1e-6
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


def test_hit_miss_and_keying(tmp_path):
    cache = SemanticCache(tmp_path / "c.db", fake_embed, threshold=0.92)
    cache.store("How do I greet people in Japan?", ["Japan"], "h1", {"answer": "bow"})

    hit = cache.lookup("how do i greet people in japan", ["japan"], "h1")
    assert hit.hit and hit.similarity >= 0.92 and hit.value == {"answer": "bow"}

    assert not cache.lookup("How do I greet people in China?", ["China"], "h1").hit  # entity key differs
    assert not cache.lookup("How do I greet people in Japan?", ["Japan"], "h2").hit  # corpus changed
    miss = cache.lookup("weather", ["Japan"], "h1")
    assert not miss.hit and miss.similarity < 0.92


def test_normalise():
    assert normalise("  How   do I GREET?! ") == "how do i greet"
