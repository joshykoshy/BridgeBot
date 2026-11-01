"""Retrieval evaluation: Recall@k, MRR, nDCG@5 over a golden set, across pipeline configs.

Concept: you can't improve what you don't measure. A golden set maps each question to the
chunk ids that contain its answer. Because chunks overlap, the same answer sentence can sit
in two neighbouring chunks; any one of them answers the question. So metrics are computed
at the *answer* level (standard top-k retrieval accuracy for single-answer QA):
  * Recall@k: 1 if any answer-bearing chunk is in the top k, else 0; averaged over
    questions (k = 1, 3, 5).
  * MRR: 1 / rank of the first answer-bearing chunk (0 if none in the top 5). Rewards
    putting the answer first, which matters because models attend most to top passages.
  * nDCG@5: log-discounted gain of the first answer-bearing chunk (ideal = rank 1).

Questions of type "unanswerable" have no relevant chunks; they are excluded from these
metrics and used for abstain accuracy instead.

Golden JSONL row: {"id", "question", "type", "relevant_ids": [...], "history": [...], "where": {...}}

Trade-off: chunk-level labels depend on the chunker; when chunking changes, the labels
must be regenerated (we generate them by searching for an answer string, then review).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

KS = (1, 3, 5)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant & set(retrieved[:k]) else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / math.log2(rank + 1)  # ideal DCG = 1/log2(2) = 1
    return 0.0


def load_golden(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("status", "approved") != "needs_review"]


def score_row(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    out = {f"R@{k}": recall_at_k(retrieved, relevant, k) for k in KS}
    out["MRR"] = mrr(retrieved, relevant)
    out["nDCG@5"] = ndcg_at_k(retrieved, relevant, 5)
    return out


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}


def evaluate_pipeline(pipeline, golden: list[dict]) -> dict:
    """Return {"overall": metrics, "by_type": {type: metrics}, "n": count, "rows": [...]}."""
    per_row, by_type = [], defaultdict(list)
    for case in golden:
        relevant = set(case.get("relevant_ids") or [])
        if not relevant:
            continue
        result = pipeline.retrieve(case["question"], history=case.get("history"), where=case.get("where"))
        retrieved = [p.id for p in result.passages]
        scores = score_row(retrieved, relevant)
        per_row.append({"id": case["id"], "type": case.get("type", ""), "retrieved": retrieved, **scores})
        by_type[case.get("type", "all")].append(scores)
    return {"overall": _mean([{k: r[k] for k in r if k not in ("id", "type", "retrieved")} for r in per_row]),
            "by_type": {t: _mean(rows) for t, rows in by_type.items()},
            "n": len(per_row), "rows": per_row}


def run_configs(pipelines: dict, golden: list[dict]) -> dict[str, dict]:
    return {name: evaluate_pipeline(p, golden) for name, p in pipelines.items()}


def markdown_table(results: dict[str, dict], key: str = "overall", type_name: str | None = None) -> str:
    cols = [f"R@{k}" for k in KS] + ["MRR", "nDCG@5"]
    lines = ["| config | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, res in results.items():
        metrics = res["by_type"].get(type_name, {}) if type_name else res[key]
        n = sum(1 for r in res["rows"] if not type_name or r["type"] == type_name)
        lines.append(f"| {name} | {n} | " + " | ".join(f"{metrics.get(c, 0):.3f}" for c in cols) + " |")
    return "\n".join(lines)
