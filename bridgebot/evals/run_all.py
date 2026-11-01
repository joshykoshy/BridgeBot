"""Run every BridgeBot eval and write data/logs/eval_report.md (+ eval_results.json).

    python -m bridgebot.evals.run_all            # everything
    python -m bridgebot.evals.run_all --quick    # skip the slow answer-level eval

Sections: retrieval table per config and question type, abstain sweep, answer quality
(groundedness, citation coverage, abstain accuracy), safety (de-escalation and false
positives), semantic cache, and agent trajectory tests. Every number in the README comes
from this file's output.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

from ..config import LOG_DIR
from ..ragcore import llm
from ..ragcore.evals import load_golden
from . import answers, retrieval
from .build_golden import GOLDEN

REPORT = LOG_DIR / "eval_report.md"
RESULTS = LOG_DIR / "eval_results.json"


def agent_section() -> tuple[dict, list[str]]:
    try:
        from .trajectories import run as run_trajectories
    except ImportError:
        return {}, ["_Agent trajectory tests not available._"]
    res = run_trajectories()
    lines = ["| test | result | detail |", "|---|---|---|"]
    lines += [f"| {t['name']} | {'PASS' if t['ok'] else 'FAIL'} | {t['detail']} |" for t in res["tests"]]
    return res, lines


def main(quick: bool = False) -> dict:
    golden = load_golden(GOLDEN)
    out: dict = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "model": llm.badge()}
    md = [f"# BridgeBot eval report", "", f"Generated {out['generated']} | brain: {out['model']}", "",
          "## Retrieval", ""]
    ret = retrieval.run()
    out["retrieval"] = {k: {"overall": v["overall"], "by_type": v["by_type"]} for k, v in ret["results"].items()}
    out["abstain_sweep"] = ret["sweep"]["rows"]
    md.append(ret["markdown"])

    if not quick:
        ans = answers.answer_evals(golden)
        out["answers"] = {k: v for k, v in ans.items() if k != "rows"}
        md += ["", "## Answers (full pipeline, cache off)", "",
               f"Judge: {ans['judge']}. n = {ans['n']} questions.", "",
               "| metric | value |", "|---|---|",
               f"| groundedness (answered questions) | {ans['groundedness']:.3f} |",
               f"| citation coverage (claim sentences with a valid [n]) | {ans['citation_coverage']:.3f} |",
               f"| citation check pass rate | {ans['citation_pass_rate']:.3f} |",
               f"| abstain accuracy (answer vs abstain correct) | {ans['abstain_accuracy']:.3f} |",
               f"| safe fallbacks used | {ans['fallbacks']} |"]

    saf = answers.safety_evals(golden)
    out["safety"] = saf
    md += ["", "## Safety", "", f"Offensive prompts de-escalated: **{saf['deescalated']}/{saf['total']}**. "
           f"False positives on {saf['benign_total']} benign questions: **{len(saf['false_positives'])}**.", ""]
    md += [f"- {'ok' if r['ok'] else 'MISSED'}: {r['prompt']}" for r in saf["offensive"]]

    cache = answers.cache_evals()
    out["cache"] = cache
    md += ["", "## Semantic cache", "", "| check | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in cache.items()]

    agent, agent_md = agent_section()
    out["agent"] = agent
    md += ["", "## Brief-me agent trajectories", "", *agent_md]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    RESULTS.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {REPORT}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    main(parser.parse_args().quick)
