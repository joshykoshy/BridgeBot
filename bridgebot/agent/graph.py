"""The Brief-me LangGraph: wiring, checkpointer and the start/get/resume service functions.

    parse_request -> check_calendar -> recall_profile -> plan_research -> retrieve_each
      -> draft_briefing -> sensitivity_check --(flags or save requested)--> human_review
                                        \\--(clean, no save)--> update_profile -> END
      human_review --(approve/edit + save)--> save_to_playbook -> update_profile -> END
                   \\--(reject, or no save requested)--> update_profile -> END

Why a graph instead of one big prompt: each step is small, testable and observable; control
flow (when to ask a human, when writing is allowed) lives in code, not in the model's
judgement; and the checkpointer makes the run pausable.

Checkpointer vs memory: the SQLite checkpointer (data/checkpoints.db) stores the *execution
state of one briefing run* (thread_id) after every node, which is what makes interrupt()
and resume possible, even after a server restart. The profile store (store.py) is
long-term memory *about the user* that outlives any single run.

recursion_limit=20 bounds the number of steps, so a bug cannot loop forever (the happy
path is 10 steps).
"""
from __future__ import annotations

import sqlite3
import uuid
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ..config import CHECKPOINT_DB, TRACE_LOG
from ..ragcore import trace
from . import nodes
from .schemas import BriefState

RECURSION_LIMIT = 20


def build_graph(checkpointer=None):
    g = StateGraph(BriefState)
    for name in ("parse_request", "check_calendar", "recall_profile", "plan_research", "retrieve_each",
                 "draft_briefing", "sensitivity_check", "human_review", "save_to_playbook", "update_profile"):
        g.add_node(name, getattr(nodes, name))
    g.add_edge(START, "parse_request")
    g.add_edge("parse_request", "check_calendar")
    g.add_edge("check_calendar", "recall_profile")
    g.add_edge("recall_profile", "plan_research")
    g.add_edge("plan_research", "retrieve_each")
    g.add_edge("retrieve_each", "draft_briefing")
    g.add_edge("draft_briefing", "sensitivity_check")
    g.add_conditional_edges("sensitivity_check", nodes.route_after_check, ["human_review", "update_profile"])
    g.add_conditional_edges("human_review", nodes.route_after_review, ["save_to_playbook", "update_profile"])
    g.add_edge("save_to_playbook", "update_profile")
    g.add_edge("update_profile", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_graph():
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    return build_graph(SqliteSaver(conn))


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def stream_run(graph, payload, thread_id: str):
    """Yield ("update", node, update) after each node, then ("interrupt", value) if the run
    paused for review. Used by the UI for live node-by-node progress.

    The tracing context is entered around each next() rather than around the whole loop:
    a web server may resume this generator on a different thread/context after each
    yield, and a ContextVar must be reset in the context that set it."""
    tracer = trace.Tracer(TRACE_LOG, kind="brief", request_id=thread_id)
    chunks = graph.stream(payload, _config(thread_id), stream_mode="updates")
    while True:
        with trace.tracing(tracer):  # nodes run inside next(), so they see the tracer
            chunk = next(chunks, None)
        if chunk is None:
            return
        for node, update in chunk.items():
            if node == "__interrupt__":
                yield ("interrupt", None, update[0].value)
            else:
                yield ("update", node, update)


def _run(graph, payload, thread_id: str) -> dict:
    updates, pending = [], None
    for kind, node, value in stream_run(graph, payload, thread_id):
        if kind == "interrupt":
            pending = value
        else:
            updates.append({"node": node, "update": value})
    return {**snapshot(graph, thread_id), "updates": updates, "interrupt": pending}


def new_thread_id() -> str:
    return f"brief-{uuid.uuid4().hex[:10]}"


def initial_state(request: str, user_id: str, save_to_playbook: bool) -> dict:
    return {"request": request, "user_id": user_id, "save_to_playbook": save_to_playbook, "events": []}


def snapshot(graph, thread_id: str) -> dict:
    state = graph.get_state(_config(thread_id))
    values = dict(state.values or {})
    waiting = list(state.next or [])
    status = "awaiting_review" if "human_review" in waiting else values.get("status", "unknown")
    return {"thread_id": thread_id, "status": status, "next": waiting, "state": values}


def start_brief(request: str, user_id: str = "demo", save_to_playbook: bool = False, graph=None) -> dict:
    graph = graph or get_graph()
    thread_id = new_thread_id()
    return _run(graph, initial_state(request, user_id, save_to_playbook), thread_id)


def get_brief(thread_id: str, graph=None) -> dict:
    return snapshot(graph or get_graph(), thread_id)


def resume_brief(thread_id: str, decision: str, edited_text: str = "", graph=None) -> dict:
    graph = graph or get_graph()
    if "human_review" not in (graph.get_state(_config(thread_id)).next or ()):
        raise ValueError(f"thread {thread_id} is not waiting for review")
    return _run(graph, Command(resume={"decision": decision, "edited_text": edited_text}), thread_id)
