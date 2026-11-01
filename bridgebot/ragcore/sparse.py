"""Sparse (lexical) retrieval with BM25.

Concept: BM25 scores a passage by how many query terms it contains, weighting rare terms
higher (IDF) and saturating repeated terms (k1) with length normalisation (b). It is the
classic search-engine ranking function and is excellent at exact matches: names
("Chuseok"), hyphenated terms ("Eid al-Adha"), and jargon the embedding model never saw.

The tokenizer keeps hyphenated/apostrophe terms intact ("al-adha") *and* emits their
parts ("al", "adha"), so both "al-Adha" and "al Adha" match. A small stop-word list
removes words that carry no signal.

Trade-off: BM25 has no notion of synonyms or paraphrase; it is in-memory and rebuilt at
start-up (well under a second at this corpus size), which is fine here but would move to a
search engine (e.g. OpenSearch) for millions of passages.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'’][a-z0-9]+)*")
SPLIT_RE = re.compile(r"[-'’]")
STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i in is it of on or should the to what when "
    "where which who why will with you your my me we our can there this that these those".split())


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        parts = SPLIT_RE.split(tok)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p and p not in STOPWORDS)
    return tokens


def matches(meta: dict, where: dict | None) -> bool:
    """Same filter semantics as dense.to_chroma_where: equality, or membership for lists."""
    if not where:
        return True
    return all(meta.get(k) in v if isinstance(v, list) else meta.get(k) == v for k, v in where.items())


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        self.ids = ids
        self.metadatas = metadatas
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 20, where: dict | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = [(self.ids[i], float(scores[i])) for i in order
                if scores[i] > 0 and matches(self.metadatas[i], where)]
        return hits[:k]
