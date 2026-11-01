"""v1 had one global chat history list shared by every user; v2 isolates sessions."""
from bridgebot import store


def test_history_isolated_per_session(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.db")
    store.add_turn("alice", "user", "Tell me about Diwali")
    store.add_turn("bob", "user", "Tell me about Hanukkah")
    assert [t["content"] for t in store.get_history("alice")] == ["Tell me about Diwali"]
    assert [t["content"] for t in store.get_history("bob")] == ["Tell me about Hanukkah"]
    for i in range(10):
        store.add_turn("alice", "user", f"q{i}")
    assert len(store.get_history("alice", turns=6)) == 6  # only recent turns go to the model


def test_profile_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STATE_DB", tmp_path / "state.db")
    store.save_profile({"user_id": "u1", "home_culture": "India", "dietary_needs": "vegetarian",
                        "past_trips": [{"location": "Japan"}], "notes": ""})
    p = store.get_profile("u1")
    assert p["dietary_needs"] == "vegetarian" and p["past_trips"] == [{"location": "Japan"}]
