"""v1 indexed one page; v2 has a cached, attributed corpus with metadata on every chunk."""
from bridgebot.config import CORPUS_DIR
from bridgebot.corpus import canonical_entities, load_chunks, parse_front_matter


def test_corpus_has_60_plus_sources_with_metadata():
    files = sorted(CORPUS_DIR.glob("*.md"))
    assert len(files) >= 60
    for path in files:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        assert meta["source_url"].startswith("http") and meta["title"] and meta["retrieved_at"]
        assert {"country", "religion", "topic"} <= meta.keys()
        assert len(body.split()) > 100


def test_chunks_carry_citation_metadata():
    chunks = load_chunks()
    assert len(chunks) > 1000
    assert all(c.metadata["source_url"] and c.metadata["section"] for c in chunks)


def test_entities_are_canonical():
    assert canonical_entities("Japanese gift etiquette") == canonical_entities("gift etiquette in Japan") == ["Japan"]
