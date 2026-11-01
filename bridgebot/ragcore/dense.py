"""Dense (semantic) retrieval: bge-small embeddings stored in a persistent Chroma collection.

Concept: an embedding model maps text to a 384-d vector so that passages with similar
*meaning* are close, even with no shared words ("greeting" ~ "bowing"). We use
BAAI/bge-small-en-v1.5: small enough for CPU, strong on retrieval benchmarks.

Details worth explaining:
  * Vectors are L2-normalised, so cosine similarity == dot product.
  * bge was trained with an instruction prefix on the *query* side only
    ("Represent this sentence for searching relevant passages: "). Passages get no prefix.
  * Chroma's PersistentClient stores vectors on disk. The collection records a corpus hash;
    if the corpus has not changed we reuse the index instead of re-embedding on every start
    (v1 rebuilt an in-memory store each launch).
  * Metadata filters (e.g. {"country": "Japan"}) are pushed down into Chroma.

Trade-off: dense retrieval is weak on exact names, rare terms and numbers; that's why it is
paired with BM25 and fused with RRF.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Return L2-normalised embeddings, shape (n, 384)."""
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    return _model().encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def to_chroma_where(where: dict | None) -> dict | None:
    """Convert {"k": v, "k2": [a, b]} into Chroma's filter syntax."""
    if not where:
        return None
    clauses = [{k: {"$in": v}} if isinstance(v, list) else {k: v} for k, v in where.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _clean_meta(meta: dict) -> dict:
    """Chroma accepts only str/int/float/bool metadata values."""
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


class DenseIndex:
    def __init__(self, persist_dir: str | Path, collection: str = "corpus"):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.name = collection
        self.collection = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def corpus_hash(self) -> str | None:
        return (self.collection.metadata or {}).get("corpus_hash")

    def build(self, ids: list[str], texts: list[str], metadatas: list[dict], corpus_hash: str) -> None:
        """(Re)index everything. Called only when the corpus hash changed."""
        self.client.delete_collection(self.name)
        self.collection = self.client.create_collection(
            self.name, metadata={"hnsw:space": "cosine", "corpus_hash": corpus_hash})
        vectors = embed(texts)
        for start in range(0, len(ids), 500):
            end = start + 500
            self.collection.add(
                ids=ids[start:end], documents=texts[start:end],
                embeddings=vectors[start:end].tolist(),
                metadatas=[_clean_meta(m) for m in metadatas[start:end]])

    def search(self, query: str, k: int = 20, where: dict | None = None) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine_similarity) best first."""
        if self.collection.count() == 0:
            return []
        vector = embed([query], is_query=True)[0].tolist()
        res = self.collection.query(query_embeddings=[vector], n_results=min(k, self.collection.count()),
                                    where=to_chroma_where(where))
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]
