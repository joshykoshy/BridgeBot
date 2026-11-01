"""Brief-me agent: trajectory tests (offline) + deterministic node helpers."""
import datetime as dt
import os

import pytest

from bridgebot.agent.nodes import observance_relevant, parse_date, rule_flags

os.environ.setdefault("OFFLINE", "1")


def test_parse_date_formats():
    today = dt.date(2026, 11, 1)
    assert parse_date("lunch on 2026-03-05", today) == "2026-03-05"
    assert parse_date("dinner on 8 November 2026", today) == "2026-11-08"
    assert parse_date("meeting March 5, 2027", today) == "2027-03-05"
    assert parse_date("on 5 March", today) == "2027-03-05"  # past date this year -> next year
    assert parse_date("no date here", today) == ""


def test_observance_relevance_uses_word_boundaries():
    holi = {"name": "Holi", "religion": "Hinduism"}
    ramadan = {"name": "Ramadan", "religion": "Islam"}
    saudi = ["Eid al-Fitr Holiday", "National Day Holiday"]
    assert not observance_relevant(holi, saudi, [])  # 'holi' inside 'Holiday' must not count
    assert observance_relevant(ramadan, saudi, [])   # Eid al-Fitr is a holiday there -> Ramadan matters
    assert observance_relevant(holi, ["Holi"], [])
    assert observance_relevant(holi, [], ["Hinduism"])  # religion named in the request


def test_rule_flags_generalisation_and_uncited():
    draft = "## Greetings\n- All Japanese people always bow deeply.\n- Bowing is common in Japan [1]."
    kinds = {f["kind"] for f in rule_flags(draft, 1)}
    assert "generalisation" in kinds and "uncited_claim" in kinds


@pytest.mark.slow
def test_trajectories():
    from bridgebot.evals.trajectories import scenarios
    results = scenarios()
    failed = [r for r in results if not r["ok"]]
    assert not failed, failed
