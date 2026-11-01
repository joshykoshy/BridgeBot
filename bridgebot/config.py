"""Paths and tunables in one place, so nothing else hard-codes a location."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"
INDEX_DIR = DATA / "index"
LOG_DIR = DATA / "logs"
CALENDAR_DB = DATA / "calendar.db"
STATE_DB = DATA / "state.db"          # chat history, profiles, playbook
CACHE_DB = DATA / "cache.db"          # semantic cache
CHECKPOINT_DB = DATA / "checkpoints.db"  # LangGraph checkpointer
TRACE_LOG = LOG_DIR / "trace.jsonl"
ASK_LOG = LOG_DIR / "ask_log.jsonl"

ABSTAIN_THRESHOLD = 0.01  # reranker probability below which we abstain; picked by the sweep in the eval report
CACHE_THRESHOLD = 0.92
HISTORY_TURNS = 6
