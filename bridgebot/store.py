"""Application state in SQLite: per-session chat history, user profiles, team playbook.

Two kinds of memory, deliberately separate:
  * short-term: `chat_history` keyed by session_id. v1 kept ONE global Python list, so every
    user saw every other user's conversation and it grew forever. Now each session has its
    own history and only the last few turns are sent to the model.
  * long-term: `profiles` (home culture, dietary needs, past trips) survive across sessions
    and are read by the Brief-me agent; `playbook` holds human-approved briefings.
(The LangGraph checkpointer is a third, different thing: it stores the *execution state*
of one agent run so it can pause for review and resume. See ARCHITECTURE.md.)

Trade-off: SQLite is a single file with no server, ideal for a demo and a single process;
a multi-instance deployment would move this to Postgres.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing

from .config import STATE_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_history (session_id TEXT, role TEXT, content TEXT, ts REAL);
CREATE INDEX IF NOT EXISTS ix_hist ON chat_history(session_id, ts);
CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, home_culture TEXT, dietary_needs TEXT,
    past_trips TEXT, notes TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS playbook (id INTEGER PRIMARY KEY, thread_id TEXT, user_id TEXT, title TEXT,
    cultures TEXT, briefing TEXT, decision TEXT, created REAL);
"""


def connect() -> sqlite3.Connection:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE_DB)
    conn.executescript(SCHEMA)
    return conn


# --- short-term memory ---------------------------------------------------------------------
def get_history(session_id: str, turns: int = 6) -> list[dict]:
    with closing(connect()) as conn:
        rows = conn.execute("SELECT role, content FROM chat_history WHERE session_id=? ORDER BY ts DESC LIMIT ?",
                            (session_id, turns)).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def add_turn(session_id: str, role: str, content: str) -> None:
    with closing(connect()) as conn, conn:
        conn.execute("INSERT INTO chat_history VALUES (?,?,?,?)", (session_id, role, content, time.time()))


def clear_history(session_id: str) -> None:
    with closing(connect()) as conn, conn:
        conn.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))


# --- long-term memory ----------------------------------------------------------------------
def get_profile(user_id: str) -> dict:
    with closing(connect()) as conn:
        row = conn.execute("SELECT home_culture, dietary_needs, past_trips, notes FROM profiles WHERE user_id=?",
                           (user_id,)).fetchone()
    if not row:
        return {"user_id": user_id, "home_culture": "", "dietary_needs": "", "past_trips": [], "notes": ""}
    return {"user_id": user_id, "home_culture": row[0], "dietary_needs": row[1],
            "past_trips": json.loads(row[2] or "[]"), "notes": row[3] or ""}


def save_profile(profile: dict) -> None:
    with closing(connect()) as conn, conn:
        conn.execute("INSERT OR REPLACE INTO profiles VALUES (?,?,?,?,?,?)",
                     (profile["user_id"], profile.get("home_culture", ""), profile.get("dietary_needs", ""),
                      json.dumps(profile.get("past_trips", [])), profile.get("notes", ""), time.time()))


# --- team playbook (written only by the agent's save_to_playbook node) ---------------------
def add_playbook_entry(thread_id: str, user_id: str, title: str, cultures: list[str], briefing: str,
                       decision: str) -> int:
    with closing(connect()) as conn, conn:
        cur = conn.execute("INSERT INTO playbook(thread_id, user_id, title, cultures, briefing, decision, created) "
                           "VALUES (?,?,?,?,?,?,?)",
                           (thread_id, user_id, title, json.dumps(cultures), briefing, decision, time.time()))
        return int(cur.lastrowid)


def playbook_entries(thread_id: str | None = None) -> list[dict]:
    sql = "SELECT id, thread_id, user_id, title, cultures, decision, created FROM playbook"
    args: tuple = ()
    if thread_id:
        sql += " WHERE thread_id=?"
        args = (thread_id,)
    with closing(connect()) as conn:
        rows = conn.execute(sql + " ORDER BY created DESC", args).fetchall()
    keys = ["id", "thread_id", "user_id", "title", "cultures", "decision", "created"]
    return [dict(zip(keys, r)) for r in rows]
