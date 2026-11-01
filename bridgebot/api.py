"""FastAPI service: a thin HTTP adapter over the same functions the UI and CLI use.

    uvicorn bridgebot.api:app --port 8000      (docs at http://localhost:8000/docs)

Endpoints: POST /ask, POST /ask/feedback, POST /brief/start, GET /brief/{id},
POST /brief/{id}/resume, GET /brief/graph, GET|PUT /profile/{user_id}, GET /evals/latest,
GET /health. Request bodies are Pydantic models, so bad input gets a 422 before any work.
"""
from __future__ import annotations

import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import store
from .agent import get_brief, mermaid, resume_brief, start_brief
from .ask import ask, log_feedback
from .config import LOG_DIR
from .corpus import get_indexes
from .ragcore import llm

app = FastAPI(title="BridgeBot v2", version="2.0",
              description="Cited cultural-intelligence answers (Ask) and reviewed meeting briefings (Brief me).")


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    use_dense: bool = True
    use_bm25: bool = True
    use_rerank: bool = True
    use_rewrite: bool = True
    use_cache: bool = True


class FeedbackIn(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: list[dict] = []
    rating: Literal["up", "down"]


class BriefIn(BaseModel):
    request: str = Field(min_length=5, max_length=2000)
    user_id: str = "demo"
    save_to_playbook: bool = False


class ResumeIn(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    edited_text: str = ""


class ProfileIn(BaseModel):
    home_culture: str = ""
    dietary_needs: str = ""
    past_trips: list[dict] = []
    notes: str = ""


@app.get("/health")
def health() -> dict:
    ix = get_indexes()
    return {"ok": True, "model": llm.badge(), "offline": llm.is_offline(), "chunks": len(ix.chunks),
            "corpus_hash": ix.hash}


@app.post("/ask")
async def ask_endpoint(body: AskIn) -> dict:
    res = await run_in_threadpool(ask, body.question, body.session_id, body.use_dense, body.use_bm25,
                                  body.use_rerank, body.use_rewrite, body.use_cache)
    return res.to_dict()


@app.post("/ask/feedback")
def feedback(body: FeedbackIn) -> dict:
    log_feedback(body.request_id, body.question, body.answer, body.citations, body.rating)
    return {"ok": True}


@app.post("/brief/start")
async def brief_start(body: BriefIn) -> dict:
    return await run_in_threadpool(start_brief, body.request, body.user_id, body.save_to_playbook)


@app.get("/brief/graph")
def brief_graph() -> dict:
    return {"mermaid": mermaid()}


@app.get("/brief/{thread_id}")
def brief_get(thread_id: str) -> dict:
    snap = get_brief(thread_id)
    if not snap["state"]:
        raise HTTPException(404, "unknown briefing id")
    return snap


@app.post("/brief/{thread_id}/resume")
async def brief_resume(thread_id: str, body: ResumeIn) -> dict:
    try:
        return await run_in_threadpool(resume_brief, thread_id, body.decision, body.edited_text)
    except ValueError as err:
        raise HTTPException(409, str(err)) from err


@app.get("/profile/{user_id}")
def profile_get(user_id: str) -> dict:
    return store.get_profile(user_id)


@app.put("/profile/{user_id}")
def profile_put(user_id: str, body: ProfileIn) -> dict:
    store.save_profile({"user_id": user_id, **body.model_dump()})
    return store.get_profile(user_id)


@app.get("/evals/latest")
def evals_latest() -> dict:
    results, report = LOG_DIR / "eval_results.json", LOG_DIR / "eval_report.md"
    if not report.exists():
        raise HTTPException(404, "no eval report yet: run python -m bridgebot.evals.run_all")
    data = json.loads(results.read_text(encoding="utf-8")) if results.exists() else {}
    return {"markdown": report.read_text(encoding="utf-8"), "results": data}
