"""Per-request tracing to JSONL.

Concept: observability for an LLM system means being able to answer "what happened on
request X, how long did each step take, how many tokens did it cost, did it hit the cache,
and what did it decide?". Each request gets a Tracer; each pipeline/agent step is wrapped
in `with tracer.step("rerank") as rec:` and one JSON line is appended per step:

  {"request_id", "kind", "step", "latency_ms", "tokens", "cache_hit", "decision", "ts"}

LLM calls anywhere in the call stack add their token usage to the innermost open step via
a ContextVar, so business code doesn't have to thread a tracer through every function.

Trade-off: a local JSONL file is the simplest thing that works and is greppable; in
production you would ship the same records to OpenTelemetry / a tracing backend.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

_current: ContextVar["Tracer | None"] = ContextVar("tracer", default=None)
_write_lock = threading.Lock()


class Tracer:
    def __init__(self, path: str | Path | None, kind: str = "request", request_id: str | None = None):
        self.path = Path(path) if path else None
        self.kind = kind
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self.records: list[dict] = []
        self._open: list[dict] = []

    @contextmanager
    def step(self, name: str) -> Iterator[dict]:
        rec = {"request_id": self.request_id, "kind": self.kind, "step": name,
               "tokens": 0, "cache_hit": None, "decision": None}
        self._open.append(rec)
        start = time.perf_counter()
        try:
            yield rec
        finally:
            rec["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
            rec["ts"] = time.time()
            self._open.remove(rec)
            self.records.append(rec)
            self._write(rec)

    def add_tokens(self, n: int) -> None:
        if self._open:
            self._open[-1]["tokens"] += int(n or 0)

    def total_tokens(self) -> int:
        return sum(r["tokens"] for r in self.records)

    def _write(self, rec: dict) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


@contextmanager
def tracing(tracer: Tracer) -> Iterator[Tracer]:
    """Make `tracer` the current tracer for everything called inside the block."""
    token = _current.set(tracer)
    try:
        yield tracer
    finally:
        _current.reset(token)


def current() -> Tracer | None:
    return _current.get()


def add_tokens(n: int) -> None:
    tracer = _current.get()
    if tracer:
        tracer.add_tokens(n)


@contextmanager
def step(name: str) -> Iterator[dict]:
    """Step on the current tracer, or a throwaway record if there is none."""
    tracer = _current.get()
    if tracer is None:
        yield {}
        return
    with tracer.step(name) as rec:
        yield rec
