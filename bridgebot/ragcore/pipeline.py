"""The retrieval pipeline: rewrite -> BM25 + dense -> RRF -> cross-encoder rerank -> abstain.

Concept (hybrid retrieval): lexical and semantic retrievers fail in different ways, so we
run both on every query variant, fuse all ranked lists with Reciprocal Rank Fusion, rerank
the top 20 with a cross-encoder and keep the top 5. If the best reranked score is below
`abstain_threshold`, we report `abstain=True` so the answer step can say "my sources don't
cover this" rather than hallucinate.

Every stage is a switch (`use_dense`, `use_bm25`, `use_rerank`, `use_rewrite`), which is
what lets the eval harness compare dense / bm25 / hybrid / hybrid+rerank / +rewrite on the
same golden set, and lets the UI show what each stage contributed. The `debug` dict records
queries, the top-10 of each retriever, the fused list, reranked scores and per-stage latency.

`Indexes` is built once per corpus and shared by all pipeline configs. It persists the dense
index in Chroma and skips re-embedding when the corpus hash is unchanged.

Trade-off: running 3 queries x 2 retrievers + a cross-encoder adds latency (roughly
0.3-1.5 s on CPU) compared with a single vector lookup; the evals show what it buys.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import trace
from .chunker import Chunk
from .dense import DenseIndex
from .fusion import reciprocal_rank_fusion
from .rerank import rerank
from .rewrite import rewrite
from .sparse import BM25Index

CANDIDATES = 20
TOP_K = 5
DEBUG_TOP = 10


def corpus_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


class Indexes:
    """BM25 (in memory) + dense (persistent Chroma) over one list of chunks."""

    def __init__(self, chunks: list[Chunk], persist_dir: str | Path, collection: str = "corpus"):
        self.chunks = {c.id: c for c in chunks}
        self.hash = corpus_hash(chunks)
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        metas = [c.metadata for c in chunks]
        self.bm25 = BM25Index(ids, texts, metas)
        self.dense = DenseIndex(persist_dir, collection)
        self.rebuilt = self.dense.corpus_hash() != self.hash or self.dense.collection.count() != len(chunks)
        if self.rebuilt:
            self.dense.build(ids, texts, metas, self.hash)


@dataclass
class Passage:
    n: int  # citation number, 1-based
    id: str
    text: str
    metadata: dict
    score: float

    @property
    def url(self) -> str:
        return self.metadata.get("source_url", "")


@dataclass
class RetrievalResult:
    question: str
    passages: list[Passage]
    abstain: bool
    debug: dict = field(default_factory=dict)

    @property
    def context(self) -> str:
        """Numbered source blocks for the answer prompt. Content is data, not instructions."""
        return "\n\n".join(
            f'<source id={p.n} url="{p.url}" title="{p.metadata.get("title", "")}">\n{p.text}\n</source>'
            for p in self.passages)

    @property
    def citations(self) -> list[dict]:
        return [{"n": p.n, "id": p.id, "title": p.metadata.get("title", ""), "url": p.url,
                 "score": round(p.score, 4), "text": p.text} for p in self.passages]


class RagPipeline:
    def __init__(self, indexes: Indexes, use_dense: bool = True, use_bm25: bool = True,
                 use_rerank: bool = True, use_rewrite: bool = True, abstain_threshold: float = 0.05,
                 vocab: list[str] | None = None):
        if not (use_dense or use_bm25):
            raise ValueError("enable at least one retriever")
        self.ix = indexes
        self.use_dense, self.use_bm25 = use_dense, use_bm25
        self.use_rerank, self.use_rewrite = use_rerank, use_rewrite
        self.abstain_threshold = abstain_threshold
        self.vocab = vocab

    def config(self) -> dict:
        return {"dense": self.use_dense, "bm25": self.use_bm25, "rerank": self.use_rerank,
                "rewrite": self.use_rewrite, "abstain_threshold": self.abstain_threshold}

    def retrieve(self, question: str, history: list[dict] | None = None, where: dict | None = None,
                 k: int = TOP_K) -> RetrievalResult:
        debug: dict = {"config": self.config(), "where": where, "latency_ms": {}}
        lat = debug["latency_ms"]

        # 1. Rewrite: standalone question + variants.
        t0 = time.perf_counter()
        queries = [question]
        search_query = question
        if self.use_rewrite:
            with trace.step("rewrite") as rec:
                rw, mode = rewrite(question, history, self.vocab)
                rec["decision"] = mode
            search_query = rw.standalone
            queries = list(dict.fromkeys([rw.standalone, *rw.variants]))
            debug["rewrite_mode"] = mode
        debug["queries"] = queries
        lat["rewrite"] = _ms(t0)

        # 2. Retrieve with each retriever for each query.
        lists: list[list[str]] = []
        t0 = time.perf_counter()
        if self.use_bm25:
            with trace.step("bm25"):
                bm25_runs = [self.ix.bm25.search(q, CANDIDATES, where) for q in queries]
            lists += [[i for i, _ in run] for run in bm25_runs]
            debug["bm25_top10"] = _top(bm25_runs[0])
        lat["bm25"] = _ms(t0)
        t0 = time.perf_counter()
        if self.use_dense:
            with trace.step("dense"):
                dense_runs = [self.ix.dense.search(q, CANDIDATES, where) for q in queries]
            lists += [[i for i, _ in run] for run in dense_runs]
            debug["dense_top10"] = _top(dense_runs[0])
        lat["dense"] = _ms(t0)

        # 3. Fuse.
        fused = reciprocal_rank_fusion(lists)[:CANDIDATES]
        debug["fused"] = _top(fused, CANDIDATES)

        # 4. Rerank (or take the fused order).
        t0 = time.perf_counter()
        if self.use_rerank:
            with trace.step("rerank"):
                ranked = rerank(search_query, [(cid, self.ix.chunks[cid].text) for cid, _ in fused], top_n=k)
            debug["reranked"] = _top(ranked, k)
        else:
            ranked = fused[:k]
        lat["rerank"] = _ms(t0)

        top_score = ranked[0][1] if ranked else 0.0
        abstain = not ranked or (self.use_rerank and top_score < self.abstain_threshold)
        debug.update(top_score=round(top_score, 4), abstain=abstain)
        passages = [Passage(n, cid, self.ix.chunks[cid].text, self.ix.chunks[cid].metadata, score)
                    for n, (cid, score) in enumerate(ranked, start=1)]
        return RetrievalResult(question, passages, abstain, debug)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _top(run: list[tuple[str, float]], n: int = DEBUG_TOP) -> list[dict]:
    return [{"id": i, "score": round(float(s), 4)} for i, s in run[:n]]
