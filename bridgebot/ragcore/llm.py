"""One small LLM wrapper: provider switch (OpenAI / Anthropic), two model tiers, structured
outputs, and an explicit offline mode.

Concept:
  * Model routing by task: a cheap *fast* model for rewriting, routing, judging and SQL;
    a stronger *smart* model for user-facing answers and drafting. Cost and latency scale
    with the number of calls, and most calls are simple classification/extraction.
  * Structured outputs: whenever the model returns data, we pass a Pydantic class and the
    provider constrains decoding to its JSON schema (OpenAI `response_format`, Anthropic
    `output_format`). We still validate with Pydantic and retry once on failure.
  * Offline mode (no API key, or OFFLINE=1): `is_offline()` is True and every caller uses a
    deterministic fallback, so demos and tests run without internet. Calls that fail at
    runtime raise LLMError, which callers also treat as "use the fallback".

Config (environment / .env):
  LLM_PROVIDER=openai|anthropic
  LLM_FAST_MODEL / LLM_SMART_MODEL            (OpenAI, default gpt-4o-mini / gpt-4o)
  ANTHROPIC_FAST_MODEL / ANTHROPIC_SMART_MODEL (default claude-haiku-5-5 / claude-opus-5-5)

Trade-off: a thin hand-written wrapper instead of LangChain's chat models keeps every line
explainable and avoids version churn, at the cost of re-implementing the two calls we need.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from . import trace

load_dotenv()

T = TypeVar("T", bound=BaseModel)
Tier = Literal["fast", "smart"]

DEFAULTS = {
    "openai": {"fast": "gpt-4o-mini", "smart": "gpt-4o"},
    "anthropic": {"fast": "claude-haiku-5-5", "smart": "claude-opus-5-5"},
}
ENV_KEYS = {
    "openai": {"fast": "LLM_FAST_MODEL", "smart": "LLM_SMART_MODEL", "key": "OPENAI_API_KEY"},
    "anthropic": {"fast": "ANTHROPIC_FAST_MODEL", "smart": "ANTHROPIC_SMART_MODEL", "key": "ANTHROPIC_API_KEY"},
}


class LLMError(RuntimeError):
    """Any provider failure (network, refusal, invalid output after retry)."""


class _InvalidOutput(Exception):
    """Model answered but produced no schema-valid object; worth one retry."""


def provider() -> str:
    name = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    return name if name in DEFAULTS else "openai"


def is_offline() -> bool:
    return os.getenv("OFFLINE", "0") == "1" or not os.getenv(ENV_KEYS[provider()]["key"])


def model_for(tier: Tier) -> str:
    p = provider()
    return os.getenv(ENV_KEYS[p][tier]) or DEFAULTS[p][tier]


def badge() -> str:
    """Short label for the UI: which brain answered."""
    if is_offline():
        return "Offline (deterministic fallbacks)"
    return f"{provider()}: {model_for('smart')} / {model_for('fast')}"


@lru_cache(maxsize=2)
def _client(name: str):
    if name == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    import openai
    return openai.OpenAI()


def complete(system: str, user: str, tier: Tier = "smart", max_tokens: int = 1500,
             temperature: float = 0.2) -> str:
    """Free-text completion. Raises LLMError on failure."""
    if is_offline():
        raise LLMError("offline")
    try:
        if provider() == "anthropic":
            # Current Claude models reject sampling params, so temperature is not sent.
            resp = _client("anthropic").messages.create(
                model=model_for(tier), max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}])
            if resp.stop_reason == "refusal":
                raise LLMError("model refused")
            trace.add_tokens(resp.usage.input_tokens + resp.usage.output_tokens)
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        resp = _client("openai").chat.completions.create(
            model=model_for(tier), max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        trace.add_tokens(resp.usage.total_tokens if resp.usage else 0)
        return (resp.choices[0].message.content or "").strip()
    except LLMError:
        raise
    except Exception as exc:  # network, auth, rate limit: caller falls back
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc


def _structured_once(system: str, user: str, schema: type[T], tier: Tier, max_tokens: int) -> T:
    if provider() == "anthropic":
        resp = _client("anthropic").messages.parse(
            model=model_for(tier), max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}], output_format=schema)
        trace.add_tokens(resp.usage.input_tokens + resp.usage.output_tokens)
        if resp.parsed_output is None:
            raise _InvalidOutput("no parsed output")
        return resp.parsed_output
    client = _client("openai")
    parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse
    resp = parse(model=model_for(tier), max_tokens=max_tokens, temperature=0, response_format=schema,
                 messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    trace.add_tokens(resp.usage.total_tokens if resp.usage else 0)
    parsed = resp.choices[0].message.parsed
    if parsed is None:  # refusal or empty
        raise _InvalidOutput(resp.choices[0].message.refusal or "no parsed output")
    return schema.model_validate(parsed.model_dump())


def structured(system: str, user: str, schema: type[T], tier: Tier = "fast",
               max_tokens: int = 1200, retries: int = 1) -> T:
    """Return a validated `schema` instance. Retries once on validation errors."""
    if is_offline():
        raise LLMError("offline")
    last: Exception | None = None
    for _ in range(retries + 1):
        try:
            return _structured_once(system, user, schema, tier, max_tokens)
        except (ValidationError, _InvalidOutput) as exc:
            last = exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
    raise LLMError(f"invalid structured output after retry: {last}")
