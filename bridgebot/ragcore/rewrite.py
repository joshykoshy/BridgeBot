"""Query rewriting and multi-query expansion.

Concept: users ask follow-ups like "what about gifts there?". Retrieving on that literal
text fails: "there" means Japan only in the chat history. So before retrieval we ask the
fast model for:
  * a *standalone* rewrite that resolves pronouns/ellipsis from the history, and
  * 2 *variants* (different wording of the same need), which raise recall because each
    phrasing retrieves a slightly different candidate set; RRF merges them afterwards.

Offline fallback: append the entities (capitalised names, or names from a known vocabulary)
of the last user turn that the new question does not already mention. Crude, but it fixes
the most common follow-up failure deterministically.

Trade-off: one extra fast-model call (~300-800 ms) per question, and a bad rewrite can
drift from the user's intent, which is why the original question is still what the
reranker and the answer prompt see.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from . import llm

SYSTEM = (
    "You rewrite search queries for a retrieval system. Given a chat history and a new user "
    "question, write (1) `standalone`: the question rewritten to be fully self-contained, "
    "resolving pronouns and references like 'there' or 'that festival' from the history, "
    "keeping the user's intent and named entities exactly; (2) `variants`: exactly 2 "
    "alternative phrasings of the standalone question using different keywords or synonyms. "
    "Do not answer the question."
)
SENTENCE_START_WORDS = frozenset("What How Why When Where Which Who Is Are Do Does Can Should I The A An In On And Tell".split())


class Rewrite(BaseModel):
    standalone: str = Field(description="Self-contained version of the question")
    variants: list[str] = Field(description="Exactly two alternative phrasings")


def extract_entities(text: str, vocab: list[str] | None = None) -> list[str]:
    """Names in `text`: from `vocab` if given (case-insensitive), else capitalised phrases."""
    if vocab:
        low = text.lower()
        return [v for v in vocab if re.search(r"\b" + re.escape(v.lower()) + r"\b", low)]
    phrases = re.findall(r"\b[A-Z][\w'\-]+(?:\s+(?:al-|de\s|of\s)?[A-Z][\w'\-]+)*", text)
    return list(dict.fromkeys(p for p in phrases if p not in SENTENCE_START_WORDS))


def last_user_turn(history: list[dict] | None) -> str:
    for turn in reversed(history or []):
        if turn.get("role") == "user":
            return turn.get("content", "")
    return ""


def offline_rewrite(question: str, history: list[dict] | None, vocab: list[str] | None = None) -> Rewrite:
    previous = last_user_turn(history)
    missing = [e for e in extract_entities(previous, vocab) if e.lower() not in question.lower()]
    standalone = f"{question} ({', '.join(missing)})" if missing else question
    return Rewrite(standalone=standalone, variants=[])


def rewrite(question: str, history: list[dict] | None = None, vocab: list[str] | None = None) -> tuple[Rewrite, str]:
    """Return (Rewrite, mode) where mode is 'llm' or 'offline'."""
    if llm.is_offline():
        return offline_rewrite(question, history, vocab), "offline"
    turns = "\n".join(f"{t['role']}: {t['content'][:500]}" for t in (history or [])[-6:])
    user = f"Chat history:\n{turns or '(none)'}\n\nNew question: {question}"
    try:
        result = llm.structured(SYSTEM, user, Rewrite, tier="fast", max_tokens=300)
        result.variants = [v for v in result.variants if v.strip()][:2]
        return result, "llm"
    except llm.LLMError:
        return offline_rewrite(question, history, vocab), "offline"
