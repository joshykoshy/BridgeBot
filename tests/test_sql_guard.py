"""sql_guard must block every red-team case and allow plain SELECTs (with LIMIT enforced)."""
import sqlite3

import pytest

from bridgebot.ragcore.sql_guard import run_readonly, validate_sql

ALLOWED = {"users", "submissions", "reviews"}

RED_TEAM = [
    "DELETE FROM users",
    "UPDATE users SET reputation = 999",
    "DROP TABLE users",
    "INSERT INTO users(name) VALUES ('x')",
    "SELECT * FROM users; DROP TABLE users; --",
    "SELECT 1; DROP TABLE users; --",
    "ATTACH DATABASE 'other.db' AS other",
    "PRAGMA table_info(users)",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM moderation_log",
    "SELECT * FROM users WHERE id IN (SELECT user_id FROM secrets)",
    "CREATE TABLE x (a int)",
    "ALTER TABLE users ADD COLUMN pwned int",
    "WITH t AS (SELECT * FROM api_keys) SELECT * FROM t",
    "SELECT load_extension('evil.dll')",
    "REPLACE INTO users(id) VALUES (1)",
    "VACUUM",
    "",
    "this is not sql",
]


@pytest.mark.parametrize("sql", RED_TEAM)
def test_red_team_blocked(sql):
    ok, reason, safe = validate_sql(sql, ALLOWED)
    assert not ok, f"should block: {sql!r}"
    assert safe is None and reason


def test_select_allowed_and_limit_added():
    ok, reason, safe = validate_sql("SELECT name FROM users WHERE reputation > 10", ALLOWED)
    assert ok, reason
    assert "LIMIT 200" in safe


def test_limit_clamped_and_cte_allowed():
    ok, _, safe = validate_sql("WITH top AS (SELECT * FROM users) SELECT * FROM top LIMIT 5000", ALLOWED)
    assert ok and "LIMIT 200" in safe
    ok, _, safe = validate_sql("SELECT u.name, COUNT(*) FROM users u JOIN submissions s ON s.user_id = u.id GROUP BY u.name LIMIT 10", ALLOWED)
    assert ok and "LIMIT 10" in safe


def test_run_readonly_cannot_write(tmp_path):
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE users(id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'ana')")
    cols, rows = run_readonly(db, "SELECT name FROM users LIMIT 200")
    assert cols == ["name"] and rows == [("ana",)]
    with pytest.raises(sqlite3.OperationalError):
        run_readonly(db, "DELETE FROM users")  # even if the validator were bypassed
