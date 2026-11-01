"""Safe text-to-SQL: validate model-written SQL before it touches the database.

Concept: an LLM that writes SQL is an untrusted user typing into your database. Prompting it
to "only write SELECT" is not a control. Here the control is code, in two layers:

1. `validate_sql(sql, allowed_tables)` parses the SQL with sqlglot (a real SQL parser, not
   regex) and rejects it unless:
     * it parses, and is exactly ONE statement ("SELECT 1; DROP TABLE users" is two);
     * the statement is a SELECT (CTEs/UNION of SELECTs allowed);
     * no write/DDL/admin node appears anywhere in the tree (INSERT, UPDATE, DELETE, DROP,
       ALTER, CREATE, ATTACH, PRAGMA, ...), including inside subqueries;
     * every table referenced is in the allowlist (CTE names excepted);
     * no dangerous function (load_extension, readfile, ...).
   It then forces `LIMIT 200` (adds it, or clamps a bigger one) and returns the
   re-generated SQL, so what runs is exactly what was validated.

2. `run_readonly(db_path, sql)` opens SQLite with `mode=ro` (the OS-level file handle is
   read-only), sets `PRAGMA query_only=ON`, and installs a progress handler that aborts
   queries running past a time budget (sqlite's `timeout` argument only covers lock waits).

Defence in depth: if the parser had a bug, the read-only connection still cannot write.

Trade-off: the allowlist must be maintained by hand, and a valid SELECT can still be slow
or reveal columns you'd rather hide; column-level allowlists or views are the next step.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import sqlglot
from sqlglot import exp

MAX_ROWS = 200
_FORBIDDEN_NAMES = ("Insert", "Update", "Delete", "Drop", "Alter", "AlterTable", "Create", "Command",
                    "Merge", "TruncateTable", "Attach", "Detach", "Pragma", "Transaction", "Commit",
                    "Rollback", "Set", "Use", "Copy", "LoadData", "Grant")
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NAMES if hasattr(exp, n))
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "sqlite_offset"}
FORBIDDEN_KEYWORDS = ("attach", "detach", "pragma", "vacuum", "reindex")


def validate_sql(sql: str, allowed_tables: set[str] | list[str]) -> tuple[bool, str, str | None]:
    """Return (ok, reason, safe_sql). safe_sql is None when ok is False."""
    allowed = {t.lower() for t in allowed_tables}
    if not sql or not sql.strip():
        return False, "empty query", None
    lowered = sql.lower()
    for word in FORBIDDEN_KEYWORDS:  # these may parse as opaque Commands; reject early
        if f"{word} " in lowered or lowered.strip().startswith(word):
            return False, f"forbidden keyword: {word.upper()}", None
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as err:
        return False, f"parse error: {str(err).splitlines()[0]}", None
    if len(statements) != 1:
        return False, f"expected exactly one statement, got {len(statements)}", None
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return False, f"only SELECT is allowed, got {tree.key.upper()}", None
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return False, f"forbidden operation: {node.key.upper()}", None
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS:
                return False, f"forbidden function: {name}", None
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = {t.name.lower() for t in tree.find_all(exp.Table)} - cte_names
    if not tables and "from" in lowered:
        return False, "could not resolve tables", None
    blocked = sorted(tables - allowed)
    if blocked:
        return False, f"table not allowed: {', '.join(blocked)}", None
    return True, "ok", _enforce_limit(tree).sql(dialect="sqlite")


def _enforce_limit(tree: exp.Expression) -> exp.Expression:
    limit = tree.args.get("limit")
    if limit is None:
        return tree.limit(MAX_ROWS)
    try:
        value = int(limit.expression.name)
    except (AttributeError, ValueError):
        return tree.limit(MAX_ROWS)
    return tree.limit(min(value, MAX_ROWS))


def run_readonly(db_path: str | Path, sql: str, timeout_s: float = 3.0) -> tuple[list[str], list[tuple]]:
    """Execute an already-validated SELECT on a read-only connection. Returns (columns, rows)."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        conn.execute("PRAGMA query_only=ON")
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description or []]
        return columns, cur.fetchmany(MAX_ROWS)
    finally:
        conn.close()
