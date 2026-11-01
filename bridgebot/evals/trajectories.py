"""Agent trajectory tests: does the Brief-me graph take the right path and do the right side effects?

Each scenario runs the real graph (offline-deterministic unless a key is set) against a
throwaway state DB and checkpointer, then asserts on the path and the database:
  * a Ramadan date produces a calendar flag (and Holi is NOT flagged for Saudi Arabia);
  * choosing "save to playbook" fires the interrupt;
  * reject -> nothing written; approve -> exactly one entry; edit -> the edited text is saved;
  * a clean request with no save skips review;
  * the research plan never exceeds 6 sub-questions;
  * a paused run resumes from a *new* graph + DB connection (the checkpointer survives restarts).
Used by tests/test_agent.py and by run_all (report section).
"""
from __future__ import annotations

import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .. import store
from ..agent.graph import build_graph, resume_brief, start_brief

RAMADAN_REQUEST = "Business lunch with Saudi partners in Riyadh on 2026-03-05"
JAPAN_REQUEST = "Dinner at a colleague's home in Tokyo, Japan on 8 November 2026"


@contextmanager
def sandbox():
    """Temporary state DB + SQLite checkpointer file; restores the real DB path afterwards."""
    original = store.STATE_DB
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store.STATE_DB = Path(tmp) / "state.db"
        ckpt = Path(tmp) / "ckpt.db"
        conns = []

        def new_graph():
            conn = sqlite3.connect(ckpt, check_same_thread=False)
            conns.append(conn)
            return build_graph(SqliteSaver(conn))
        try:
            yield new_graph
        finally:
            for c in conns:
                c.close()
            store.STATE_DB = original


def _flag_names(run: dict) -> list[str]:
    return [f["text"].split(":")[0] for f in run["state"].get("calendar", {}).get("flags", [])]


def scenarios() -> list[dict]:
    results = []

    def record(name, ok, detail):
        results.append({"name": name, "ok": bool(ok), "detail": detail})

    with sandbox() as new_graph:
        g = new_graph()
        run = start_brief(RAMADAN_REQUEST, save_to_playbook=False, graph=g)
        names = _flag_names(run)
        record("ramadan_date_flags_calendar", "Ramadan" in names, f"flags={names}")
        record("holi_not_flagged_for_saudi", "Holi" not in names, f"flags={names}")
        record("plan_capped_at_6", len(run["state"]["sub_questions"]) <= 6,
               f"{len(run['state']['sub_questions'])} sub-questions")
        if not run["state"].get("flags"):
            record("clean_request_skips_review", run["status"] == "done" and run["interrupt"] is None,
                   f"status={run['status']}")

        run = start_brief(JAPAN_REQUEST, save_to_playbook=True, graph=g)
        record("save_request_fires_interrupt", run["interrupt"] is not None and run["next"] == ["human_review"],
               f"next={run['next']}")
        done = resume_brief(run["thread_id"], "reject", graph=g)
        saved = store.playbook_entries(run["thread_id"])
        record("reject_saves_nothing", done["status"] == "rejected" and not saved, f"status={done['status']} entries={len(saved)}")

        run = start_brief(JAPAN_REQUEST, save_to_playbook=True, graph=g)
        done = resume_brief(run["thread_id"], "approve", graph=g)
        saved = store.playbook_entries(run["thread_id"])
        trips = store.get_profile("demo")["past_trips"]
        record("approve_saves_once_and_updates_profile",
               done["status"] == "approved" and len(saved) == 1 and len(trips) >= 1,
               f"entries={len(saved)} past_trips={len(trips)}")

        run = start_brief(JAPAN_REQUEST, save_to_playbook=True, graph=g)
        edited = "# Edited briefing\n\n- Bring a small wrapped gift and offer it with both hands [1]."
        resume_brief(run["thread_id"], "edit", edited_text=edited, graph=g)
        with closing(sqlite3.connect(store.STATE_DB)) as conn:
            text = conn.execute("SELECT briefing FROM playbook WHERE thread_id=?", (run["thread_id"],)).fetchone()
        record("edit_saves_edited_text", text is not None and text[0] == edited, "saved text matches edit")

        run = start_brief(JAPAN_REQUEST, save_to_playbook=True, graph=g)
        fresh = new_graph()  # simulates a server restart: new graph object, new DB connection
        done = resume_brief(run["thread_id"], "approve", graph=fresh)
        record("resume_after_restart", done["status"] == "approved", f"status={done['status']}")
    return results


def run() -> dict:
    tests = scenarios()
    return {"tests": tests, "passed": sum(t["ok"] for t in tests), "total": len(tests)}


if __name__ == "__main__":
    for t in run()["tests"]:
        print("PASS" if t["ok"] else "FAIL", t["name"], "-", t["detail"])
