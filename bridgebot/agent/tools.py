"""The agent's three tools: calendar lookup, knowledge-base search, playbook write.

Least privilege: each tool is a plain function with a Pydantic input model, and each graph
node imports only the tool it needs (check_calendar -> calendar_lookup, retrieve_each ->
search_knowledge_base, save_to_playbook -> write_playbook). No LLM decides to call the write
tool; it runs only on the graph path that goes through human review with "approve" or
"edit". That is the difference between "the model is told not to write" and "the model
cannot write".

calendar_lookup is pure SQL over data/calendar.db: dates are facts, not generation.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from contextlib import closing

from .. import store
from ..config import ABSTAIN_THRESHOLD, CALENDAR_DB
from ..corpus import COUNTRIES, DEMONYMS, VOCAB, get_indexes
from ..ragcore.pipeline import RagPipeline
from .schemas import CalendarQuery, PlaybookEntry, SearchQuery

COUNTRY_ALIASES = {"UAE": "United Arab Emirates", "USA": "United States", "UK": "United Kingdom",
                   "Korea": "South Korea", "Dubai": "United Arab Emirates"}


def _canonical_country(name: str) -> str:
    return COUNTRY_ALIASES.get(name, DEMONYMS.get(name.lower(), name))


def calendar_lookup(q: CalendarQuery) -> dict:
    """Holidays in the given countries and major observances within +/- window_days."""
    q = CalendarQuery.model_validate(q)
    day = dt.date.fromisoformat(q.date)
    lo, hi = (day - dt.timedelta(days=q.window_days)).isoformat(), (day + dt.timedelta(days=q.window_days)).isoformat()
    countries = sorted({_canonical_country(c) for c in q.countries})
    with closing(sqlite3.connect(f"{CALENDAR_DB.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        holidays = []
        if countries:
            marks = ",".join("?" * len(countries))
            holidays = conn.execute(
                f"SELECT country, date, name, source FROM public_holidays WHERE country IN ({marks}) "
                "AND date BETWEEN ? AND ? ORDER BY date", (*countries, lo, hi)).fetchall()
        year_holidays = []
        if countries:
            year_holidays = [r[0] for r in conn.execute(
                f"SELECT DISTINCT name FROM public_holidays WHERE country IN ({marks}) AND substr(date, 1, 4) = ?",
                (*countries, q.date[:4]))]
        observances = conn.execute(
            "SELECT name, religion, start_date, end_date, note, source_url FROM observances "
            "WHERE start_date <= ? AND end_date >= ? ORDER BY start_date", (hi, lo)).fetchall()
    return {
        "date": q.date, "weekday": day.strftime("%A"), "countries": countries,
        "holidays": [dict(zip(("country", "date", "name", "source"), r)) for r in holidays],
        "year_holiday_names": year_holidays,
        "observances": [dict(zip(("name", "religion", "start_date", "end_date", "note", "source_url"), r))
                        for r in observances],
    }


def search_knowledge_base(q: SearchQuery) -> list[dict]:
    """Hybrid RAG search. With a culture, merge a metadata-filtered search (sources tagged
    with that country/religion) and an unfiltered one (regional articles such as
    'Etiquette in Asia' carry no single country tag)."""
    q = SearchQuery.model_validate(q)
    pipe = RagPipeline(get_indexes(), use_rewrite=False, abstain_threshold=ABSTAIN_THRESHOLD, vocab=VOCAB)
    culture = _canonical_country(q.culture) if q.culture else ""
    field = "country" if culture in COUNTRIES or culture in COUNTRY_ALIASES.values() else "religion"
    runs = []
    if culture:
        runs.append(pipe.retrieve(q.query, where={field: culture}, k=q.k))
    runs.append(pipe.retrieve(q.query, k=q.k))
    seen, out = set(), []
    for run in runs:
        for p in run.passages:
            if p.id not in seen and p.score >= ABSTAIN_THRESHOLD:
                seen.add(p.id)
                out.append({"id": p.id, "text": p.text, "title": p.metadata.get("title", ""),
                            "url": p.url, "score": round(p.score, 4), "query": q.query})
    return sorted(out, key=lambda x: x["score"], reverse=True)[: q.k + 1]


def write_playbook(entry: PlaybookEntry) -> int:
    """The only write tool. Validated before writing."""
    entry = PlaybookEntry.model_validate(entry)
    return store.add_playbook_entry(entry.thread_id, entry.user_id, entry.title, entry.cultures,
                                    entry.briefing, entry.decision)
