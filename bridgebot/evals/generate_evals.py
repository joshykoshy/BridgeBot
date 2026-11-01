"""Grow the golden set from real usage and from the corpus itself.

Two sources of candidate eval cases, both written as `status: needs_review` to
bridgebot/evals/golden_candidates.jsonl (a human promotes good ones into golden_seed.jsonl;
load_golden() ignores needs_review rows, so unreviewed cases never affect the scores):

  1. thumbs-up answers from the UI (data/logs/ask_log.jsonl): the question plus the chunk
     ids that were cited become a regression case ("this used to work, keep it working");
  2. synthetic questions from random chunks: the fast model writes one question that the
     chunk answers (offline: a template from the chunk's section heading).

Why review is mandatory: synthetic questions often copy the chunk's wording (too easy for
BM25) or are ambiguous, and a thumbs-up only says the user liked the answer, not that the
retrieval was ideal.

Run: python -m bridgebot.evals.generate_evals --synthetic 10
"""
from __future__ import annotations

import argparse
import json
import random

from pydantic import BaseModel

from ..config import ASK_LOG
from ..corpus import load_chunks
from ..ragcore import llm
from .build_golden import HERE

CANDIDATES = HERE / "golden_candidates.jsonl"
SYSTEM = ("Write ONE natural question a traveller might ask that this passage answers. Do not copy long phrases "
          "from the passage; paraphrase. Also give a short distinctive answer phrase copied from the passage.")


class SyntheticCase(BaseModel):
    question: str
    answer_phrase: str


def from_feedback() -> list[dict]:
    if not ASK_LOG.exists():
        return []
    rows = [json.loads(line) for line in ASK_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"id": f"fb-{r['request_id']}", "type": "from_feedback", "question": r["question"], "history": [],
             "relevant_ids": r["citation_ids"][:2], "status": "needs_review", "source": "thumbs_up"}
            for r in rows if r.get("rating") == "up" and r.get("citation_ids")]


def synthetic(n: int, seed: int = 7) -> list[dict]:
    chunks = [c for c in load_chunks() if len(c.text.split()) > 80]
    out = []
    for c in random.Random(seed).sample(chunks, min(n, len(chunks))):
        section = c.metadata["section"].split(" > ")
        question, mode = f"What do my sources say about {section[-1].lower()} in the context of {section[0]}?", "template"
        if not llm.is_offline():
            try:
                question, mode = llm.structured(SYSTEM, c.text[:2500], SyntheticCase, tier="fast").question, "llm"
            except llm.LLMError:
                pass
        out.append({"id": f"syn-{c.id}", "type": "synthetic", "question": question, "history": [],
                    "relevant_ids": [c.id], "status": "needs_review", "source": f"synthetic-{mode}"})
    return out


def main(n_synthetic: int) -> None:
    existing = set()
    if CANDIDATES.exists():
        existing = {json.loads(line)["id"] for line in CANDIDATES.read_text(encoding="utf-8").splitlines() if line.strip()}
    new = [r for r in from_feedback() + synthetic(n_synthetic) if r["id"] not in existing]
    with CANDIDATES.open("a", encoding="utf-8") as fh:
        for row in new:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"added {len(new)} needs_review candidates to {CANDIDATES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=int, default=10)
    main(parser.parse_args().synthetic)
