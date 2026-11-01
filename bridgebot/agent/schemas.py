"""Typed contracts for the Brief-me agent: graph state, LLM outputs and tool inputs.

Why Pydantic everywhere: every boundary where data crosses from a model or a user into code
is validated. LLM outputs use these classes as structured-output schemas, so a malformed
response fails loudly (and is retried once) instead of silently corrupting the state.
Tool inputs are validated before any tool runs.

`BriefState` is the LangGraph state. Each node returns a partial dict that is merged in;
`events` uses an additive reducer so every node can append to the progress log.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# --- LLM structured outputs -----------------------------------------------------------------
class ParsedRequest(BaseModel):
    cultures: list[str] = Field(description="Countries, regions or religions of the people involved")
    location: str = Field(description="City or country where the meeting happens, or empty")
    date: str = Field(description="Meeting date as YYYY-MM-DD, or empty if not given")
    occasion: str = Field(description="e.g. business lunch, home visit, wedding, or empty")
    attendees: str = Field(description="Who will be there, in a few words, or empty")


class ResearchPlan(BaseModel):
    sub_questions: list[str] = Field(description="4 to 6 focused questions to research")


class SensitivityFlag(BaseModel):
    kind: Literal["uncited_claim", "generalisation", "religious_claim_without_source", "other"]
    text: str = Field(description="The exact sentence that is a problem")
    reason: str


class SensitivityReport(BaseModel):
    flags: list[SensitivityFlag]


# --- Tool inputs (each tool validates its input with one of these) --------------------------
class CalendarQuery(BaseModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    countries: list[str] = []
    window_days: int = Field(default=3, ge=0, le=14)


class SearchQuery(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    culture: str = ""
    k: int = Field(default=3, ge=1, le=5)


class PlaybookEntry(BaseModel):
    thread_id: str
    user_id: str
    title: str = Field(min_length=3, max_length=200)
    cultures: list[str]
    briefing: str = Field(min_length=20)
    decision: Literal["approve", "edit"]


# --- Graph state ------------------------------------------------------------------------------
class BriefState(TypedDict, total=False):
    request: str
    user_id: str
    save_to_playbook: bool
    parsed: dict
    calendar: dict
    profile: dict
    sub_questions: list[str]
    sources: list[dict]
    draft: str
    flags: list[dict]
    review: dict
    final: str
    saved_id: int | None
    status: str
    events: Annotated[list[dict], operator.add]
