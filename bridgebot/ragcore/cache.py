"""Semantic cache in SQLite.

Concept: many questions are the same question in different words ("how do I greet people
in Japan" / "greeting etiquette Japan"). An exact-string cache misses those; a semantic
cache embeds the normalised query and returns a stored answer when cosine similarity with
a previous query is >= 0.92.

The key is (query embedding, extracted entities, corpus hash):
  * entities: "gift etiquette in Japan" and "gift etiquette in China" embed very close
    together (often > 0.92), but must never share an answer. Requiring the same entity set
    turns the most dangerous false hit into a guaranteed miss.
  * corpus hash: when the knowledge base changes, old answers stop matching automatically,
    so there is no stale-cache invalidation job to forget.

Every lookup reports hit/miss and the best similarity so the UI and traces can show it.

Trade-off: the threshold trades hit rate for correctness; 0.92 is deliberately strict. The
scan is a linear numpy dot product over rows with the same key, fine for thousands of
entries; at larger scale you would put the vectors in an ANN index.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass
class CacheResult:
    hit: bool
    similarity: float
    value: dict | None = None
    matched_query: str | None = None


def normalise(query: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", query.lower())).strip()


def entity_key(entities: list[str]) -> str:
    return "|".join(sorted({e.strip().lower() for e in entities if e.strip()}))


class SemanticCache:
    def __init__(self, db_path: str | Path, embed_fn: EmbedFn, threshold: float = 0.92):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.embed_fn = embed_fn
        self.threshold = threshold
        with closing(self._conn()) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY, query TEXT, entities TEXT, corpus_hash TEXT,
                embedding BLOB, value TEXT, created REAL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_cache_key ON semantic_cache(entities, corpus_hash)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _vector(self, query: str) -> np.ndarray:
        return np.asarray(self.embed_fn([normalise(query)])[0], dtype=np.float32)

    def lookup(self, query: str, entities: list[str], corpus_hash: str) -> CacheResult:
        vec = self._vector(query)
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT query, embedding, value FROM semantic_cache WHERE entities=? AND corpus_hash=?",
                (entity_key(entities), corpus_hash)).fetchall()
        if not rows:
            return CacheResult(False, 0.0)
        matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        sims = matrix @ vec
        best = int(np.argmax(sims))
        sim = float(sims[best])
        if sim >= self.threshold:
            return CacheResult(True, sim, json.loads(rows[best][2]), rows[best][0])
        return CacheResult(False, sim)

    def store(self, query: str, entities: list[str], corpus_hash: str, value: dict) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO semantic_cache(query, entities, corpus_hash, embedding, value, created) VALUES (?,?,?,?,?,?)",
                (query, entity_key(entities), corpus_hash, self._vector(query).tobytes(),
                 json.dumps(value, default=str), time.time()))

    def clear(self) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute("DELETE FROM semantic_cache")
