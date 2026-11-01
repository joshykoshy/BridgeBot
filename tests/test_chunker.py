"""Chunker: sections never mix, size bound holds, overlap exists, metadata survives."""
from bridgebot.ragcore.chunker import chunk_markdown, estimate_tokens, split_sections

DOC = """# Etiquette in Japan

Intro paragraph about Japan.

## Gift giving

""" + " ".join(f"gift{i}" for i in range(700)) + """

## Bowing

Bowing is a common greeting.
"""


def test_sections_split_on_headings():
    paths = [p for p, _ in split_sections(DOC)]
    assert paths == [["Etiquette in Japan"], ["Etiquette in Japan", "Gift giving"], ["Etiquette in Japan", "Bowing"]]


def test_chunks_bounded_overlapping_and_tagged():
    chunks = chunk_markdown("japan", DOC, {"title": "Etiquette in Japan", "country": "Japan"})
    gift = [c for c in chunks if "Gift giving" in c.metadata["section"]]
    assert len(gift) >= 2
    for c in gift:
        body = c.text.split("\n\n", 1)[1]
        assert estimate_tokens(body) <= 400
    first_tail = gift[0].text.split()[-10:]
    assert " ".join(first_tail) in gift[1].text  # overlap carried over
    assert all(c.metadata["country"] == "Japan" for c in chunks)
    assert [c.id for c in chunks] == [f"japan#{i}" for i in range(len(chunks))]
    bowing = [c for c in chunks if c.metadata["section"].endswith("Bowing")]
    assert len(bowing) == 1 and "gift" not in bowing[0].text  # no section mixing


def test_no_chunk_is_only_the_previous_overlap():
    """Regression: a paragraph following a split long paragraph used to emit an overlap-only chunk."""
    long_a = " ".join(f"a{i}" for i in range(600))  # remainder after the first split doesn't fit beside the overlap
    long_b = " ".join(f"b{i}" for i in range(280))
    doc = "\n\n".join(["# T", long_a, long_b])  # long_b doesn't fit next to long_a's overlap tail
    chunks = chunk_markdown("t", doc, {"title": "T"})
    bodies = [c.text.split("\n\n", 1)[1] for c in chunks]
    for prev, cur in zip(bodies, bodies[1:]):
        assert not prev.endswith(cur), "chunk adds no new text"
    assert len(set(bodies)) == len(bodies)
    assert all(estimate_tokens(b) <= 400 for b in bodies)
