import json

import pytest

from retail_agent.reports import ReportStore


@pytest.fixture
def store(tmp_path):
    s = ReportStore(tmp_path / "app.sqlite")
    s.save("alice", "conv1", "Q1 revenue", "Client X grew 10%")
    s.save("alice", "conv2", "Churn", "Nothing about clients")
    s.save("bob", "conv3", "Bob's Client X report", "Client X")
    return s


def test_find_is_owner_scoped(store):
    assert [r["title"] for r in store.find("alice", text="client x")] == ["Q1 revenue"]
    assert [r["title"] for r in store.find("alice", conversation_id="conv2")] == [
        "Churn"
    ]
    assert len(store.find("alice")) == 2


def test_delete_only_own_ids(store):
    bobs = [r["id"] for r in store.find("bob")]
    assert store.delete("alice", bobs) == 0
    assert len(store.find("bob")) == 1


def test_delete_exact_snapshot(store):
    snapshot = [r["id"] for r in store.find("alice", text="client x")]
    store.save("alice", "conv1", "Late Client X report", "Client X again")
    assert store.delete("alice", snapshot) == 1
    assert [r["title"] for r in store.find("alice")] == [
        "Churn",
        "Late Client X report",
    ]


def test_delete_writes_audit(store):
    ids = [r["id"] for r in store.find("alice")]
    store.delete("alice", ids)
    rows = store.con.execute("SELECT owner, action, ids FROM audit").fetchall()
    assert [tuple(r) for r in rows] == [("alice", "delete", json.dumps(ids))]


def test_find_without_filter_returns_all_own(store):
    assert [r["title"] for r in store.find("alice")] == ["Q1 revenue", "Churn"]


def test_concurrent_tool_threads_do_not_corrupt_store(tmp_path):
    # LangGraph runs parallel tool calls in threads that share one ReportStore.
    from concurrent.futures import ThreadPoolExecutor

    store = ReportStore(tmp_path / "app.sqlite")

    def save_and_delete(i):
        rid = store.save("alice", "c", f"r{i}", "x")
        store.find("alice", text=f"r{i}")
        return store.delete("alice", [rid])

    with ThreadPoolExecutor(16) as pool:
        assert list(pool.map(save_and_delete, range(64))) == [1] * 64
    assert store.find("alice") == []


def test_find_treats_like_wildcards_literally(store):
    store.save("alice", "conv4", "Margin 100% plan", "x")
    assert [r["title"] for r in store.find("alice", text="100%")] == [
        "Margin 100% plan"
    ]
    assert store.find("alice", text="Q_ revenue") == []


def test_get_is_owner_scoped(store):
    alice_id = store.find("alice", text="Q1")[0]["id"]
    assert store.get("alice", alice_id)["body"] == "Client X grew 10%"
    assert store.get("bob", alice_id) is None
