"""Corpus loading, the shared index, and the entity vocabulary.

Concept: one place that turns `data/corpus/*.md` (front-matter + markdown) into chunks and
builds the hybrid index once per process (`get_indexes`, cached). The dense part is
persisted in `data/index/` and only re-embedded when the corpus hash changes, so restarts
are fast (v1 re-scraped and re-embedded on every launch).

It also defines the entity vocabulary (countries, demonyms, religions, occasions). Entities
are used by the semantic cache key, by the offline query rewriter, and by the Brief-me
agent's metadata filters. Demonyms map to a canonical name ("Japanese" -> "Japan") so
"Japanese gift etiquette" and "gift etiquette in Japan" share a cache key.

Trade-off: a hand-written vocabulary is predictable and testable but must be extended for
new countries; an NER model would generalise better but adds weight and non-determinism.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .config import CORPUS_DIR, INDEX_DIR
from .ragcore.chunker import Chunk, chunk_markdown
from .ragcore.pipeline import Indexes

COUNTRIES = ["Japan", "China", "South Korea", "Korea", "India", "Indonesia", "Thailand", "Singapore", "Malaysia",
             "Saudi Arabia", "United Arab Emirates", "UAE", "Dubai", "Egypt", "Turkey", "Israel", "Brazil", "Mexico",
             "Nigeria", "South Africa", "Australia", "New Zealand", "United States", "USA", "Canada",
             "United Kingdom", "UK", "Germany", "France", "Middle East", "Latin America", "Africa", "Europe", "Asia"]
DEMONYMS = {"japanese": "Japan", "chinese": "China", "korean": "South Korea", "indian": "India",
            "indonesian": "Indonesia", "thai": "Thailand", "singaporean": "Singapore", "malaysian": "Malaysia",
            "saudi": "Saudi Arabia", "emirati": "United Arab Emirates", "egyptian": "Egypt", "turkish": "Turkey",
            "israeli": "Israel", "brazilian": "Brazil", "mexican": "Mexico", "nigerian": "Nigeria",
            "south african": "South Africa", "australian": "Australia", "american": "United States",
            "canadian": "Canada", "british": "United Kingdom", "german": "Germany", "french": "France",
            "arab": "Middle East", "latin american": "Latin America", "african": "Africa", "european": "Europe",
            "korea": "South Korea", "uae": "United Arab Emirates", "dubai": "United Arab Emirates",
            "usa": "United States", "uk": "United Kingdom"}
RELIGIONS = {"islam": "Islam", "muslim": "Islam", "muslims": "Islam", "hinduism": "Hinduism", "hindu": "Hinduism",
             "hindus": "Hinduism", "judaism": "Judaism", "jewish": "Judaism", "jews": "Judaism",
             "christianity": "Christianity", "christian": "Christianity", "christians": "Christianity",
             "buddhism": "Buddhism", "buddhist": "Buddhism", "buddhists": "Buddhism", "sikhism": "Sikhism",
             "sikh": "Sikhism", "sikhs": "Sikhism"}
OCCASIONS = ["Ramadan", "Iftar", "Eid al-Fitr", "Eid al-Adha", "Eid", "Diwali", "Holi", "Navaratri",
             "Chinese New Year", "Lunar New Year", "Seollal", "Chuseok", "Mid-Autumn Festival", "Hanukkah",
             "Shabbat", "Passover", "Yom Kippur", "Christmas", "Lent", "Easter", "Rio Carnival", "Carnival",
             "Day of the Dead", "Vesak", "Songkran", "tea ceremony", "wedding", "funeral", "business meeting",
             "dinner"]
VOCAB = COUNTRIES + sorted({*DEMONYMS, *RELIGIONS}) + OCCASIONS


def canonical_entities(text: str) -> list[str]:
    """Canonical entity names mentioned in text (countries, religions, occasions)."""
    low = text.lower()
    found: set[str] = set()
    for term in sorted(VOCAB, key=len, reverse=True):
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", low):
            found.add(DEMONYMS.get(term.lower()) or RELIGIONS.get(term.lower()) or term)
    if "South Korea" in found:
        found.discard("Korea")
    return sorted(found)


def parse_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body.lstrip()


def load_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        keep = {k: meta.get(k, "") for k in ("title", "source_url", "retrieved_at", "country", "religion", "topic")}
        chunks += chunk_markdown(path.stem, body, keep)
    return chunks


@lru_cache(maxsize=1)
def get_indexes() -> Indexes:
    return Indexes(load_chunks(), INDEX_DIR, collection="bridgebot")
