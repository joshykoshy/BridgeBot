"""Map golden questions to chunk ids by searching for an answer string.

Why: chunk ids change whenever the chunker changes, but the *answer text* doesn't. Each seed
row carries a regex for the sentence that answers it (and the documents it may come from).
This script finds every chunk containing that sentence and writes golden.jsonl with
`relevant_ids`. Rows that match nothing, or suspiciously many chunks, are printed for
manual review and marked `needs_review` (excluded from scoring until fixed).

Run: python -m bridgebot.evals.build_golden
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..corpus import load_chunks

HERE = Path(__file__).parent
SEED = HERE / "golden_seed.jsonl"
GOLDEN = HERE / "golden.jsonl"
MAX_MATCHES = 6


def build(seed: Path = SEED, out: Path = GOLDEN) -> list[dict]:
    chunks = load_chunks()
    rows = [json.loads(line) for line in seed.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row.setdefault("history", [])
        if row["type"] == "unanswerable":
            row.update(relevant_ids=[], status="approved")
            continue
        rx = re.compile(row["pattern"], re.IGNORECASE)
        docs = row.get("docs")
        hits = [c.id for c in chunks if rx.search(c.text) and (not docs or c.metadata["doc_id"] in docs)]
        row["relevant_ids"] = hits
        row["status"] = "approved" if 0 < len(hits) <= MAX_MATCHES else "needs_review"
        flag = "" if row["status"] == "approved" else "   <-- REVIEW"
        print(f"{row['id']:5} {len(hits):2} {hits}{flag}")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {out} ({sum(r['status'] == 'approved' for r in rows)}/{len(rows)} approved)")
    return rows


if __name__ == "__main__":
    build()
