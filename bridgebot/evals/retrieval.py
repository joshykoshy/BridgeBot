"""Retrieval evals: compare pipeline configurations on the golden set, and tune abstention.

Configs (each adds one idea to the previous, so the table shows what each idea buys):
  dense            bge-small vectors only (closest to v1, with better chunks)
  bm25             lexical only
  hybrid           BM25 + dense fused with RRF
  hybrid+rerank    + cross-encoder over the top 20
  +rewrite         + standalone rewrite and multi-query (LLM when a key is set, else the
                   offline entity-append rewrite; the report says which ran)

Abstain threshold: we collect the top rerank score for answerable vs unanswerable questions
and pick the threshold that maximises abstain accuracy; the README reports the sweep.
"""
from __future__ import annotations

from ..config import ABSTAIN_THRESHOLD
from ..corpus import VOCAB, get_indexes
from ..ragcore import llm
from ..ragcore.evals import load_golden, markdown_table, run_configs
from ..ragcore.pipeline import RagPipeline
from .build_golden import GOLDEN

TYPES = ("exact_name", "paraphrase", "follow_up")


def pipelines() -> dict[str, RagPipeline]:
    ix = get_indexes()
    mk = lambda **kw: RagPipeline(ix, vocab=VOCAB, abstain_threshold=ABSTAIN_THRESHOLD, **kw)  # noqa: E731
    return {
        "dense": mk(use_dense=True, use_bm25=False, use_rerank=False, use_rewrite=False),
        "bm25": mk(use_dense=False, use_bm25=True, use_rerank=False, use_rewrite=False),
        "hybrid": mk(use_rerank=False, use_rewrite=False),
        "hybrid+rerank": mk(use_rerank=True, use_rewrite=False),
        "+rewrite": mk(use_rerank=True, use_rewrite=True),
    }


def abstain_sweep(golden: list[dict], pipeline: RagPipeline) -> dict:
    scores = []
    for case in golden:
        res = pipeline.retrieve(case["question"], history=case.get("history"))
        scores.append((case["type"] == "unanswerable", res.debug["top_score"]))
    rows = []
    for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        correct = sum((s < t) == unans for unans, s in scores)
        rows.append({"threshold": t, "accuracy": correct / len(scores),
                     "false_abstain": sum(s < t and not u for u, s in scores),
                     "missed_abstain": sum(s >= t and u for u, s in scores)})
    return {"rows": rows, "scores": scores}


def run() -> dict:
    golden = load_golden(GOLDEN)
    pipes = pipelines()
    results = run_configs(pipes, golden)
    sweep = abstain_sweep(golden, pipes["+rewrite"])
    rewrite_mode = "offline entity-append" if llm.is_offline() else f"LLM ({llm.model_for('fast')})"
    md = [f"Golden set: {len(golden)} questions ({sum(g['type'] == 'unanswerable' for g in golden)} unanswerable, "
          f"excluded from retrieval metrics). Rewrite mode: {rewrite_mode}.", "",
          "**Overall**", "", markdown_table(results)]
    for t in TYPES:
        md += ["", f"**{t}**", "", markdown_table(results, type_name=t)]
    md += ["", "**Abstain threshold sweep** (top rerank score < threshold -> abstain; config +rewrite)", "",
           "| threshold | accuracy | false abstains (answerable) | missed abstains (unanswerable) |", "|---|---|---|---|"]
    md += [f"| {r['threshold']} | {r['accuracy']:.3f} | {r['false_abstain']} | {r['missed_abstain']} |" for r in sweep["rows"]]
    return {"results": results, "sweep": sweep, "markdown": "\n".join(md), "rewrite_mode": rewrite_mode}


if __name__ == "__main__":
    print(run()["markdown"])
