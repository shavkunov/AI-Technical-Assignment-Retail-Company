"""Langfuse gets one trace per user turn: question, guardrail, every model and tool
call and the final answer. Offline: spans go to memory."""

import json

import pytest
from langchain_core.messages import AIMessage
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from conftest import FakeToolModel
from retail_agent import observability
from retail_agent.agent import UserCtx, build_agent
from retail_agent.cli import run_turn
from retail_agent.reports import ReportStore

ALICE = UserCtx("alice", "brand_exec", ["Calvin Klein"], "conv-1")
COUNT = {
    "name": "run_sql",
    "args": {"sql": "SELECT COUNT(*) AS n FROM orders"},
    "id": "s1",
}


@pytest.fixture(scope="module")
def client():
    # Langfuse keeps one client per public key: create it once, clear spans per test
    exporter = InMemorySpanExporter()
    return exporter, Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="http://localhost:1",
        span_exporter=exporter,
    )


@pytest.fixture
def spans(client, monkeypatch):
    exporter, langfuse = client
    exporter.clear()
    monkeypatch.setattr(observability, "langfuse_client", lambda: langfuse)
    return exporter


def attr(span, key):
    return span.attributes.get(key)


def turn(local_db, tmp_path, script, question="how many orders?"):
    agent = build_agent(
        FakeToolModel(messages=iter(script)),
        local_db,
        ReportStore(tmp_path / "app.sqlite"),
        prefs_path=tmp_path / "p.json",
    )
    run_turn(agent, ALICE, question, lambda r: True, tmp_path / "t.jsonl")


def test_turn_is_one_langfuse_trace(local_db, tmp_path, spans):
    script = [AIMessage("", tool_calls=[COUNT]), AIMessage("Done.")]
    turn(local_db, tmp_path, script)

    finished = spans.get_finished_spans()
    assert len({s.context.trace_id for s in finished}) == 1  # one trace per turn
    [root] = [s for s in finished if s.parent is None]
    assert root.name == "turn"
    assert attr(root, "user.id") == "alice"
    assert attr(root, "session.id") == "conv-1"
    assert "how many orders?" in attr(root, "langfuse.observation.input")
    assert "Done." in attr(root, "langfuse.observation.output")
    meta = json.dumps(dict(root.attributes))
    assert "ANALYTICS" in meta  # guardrail label
    assert "sql_errors" in meta

    names = [s.name for s in finished]
    assert "run_sql" in names
    assert any("input_guardrail" in n for n in names)


def test_failed_turn_is_an_error_trace(local_db, tmp_path, spans):
    turn(local_db, tmp_path, [], "revenue?")  # empty script: the agent's model fails
    [root] = [s for s in spans.get_finished_spans() if s.parent is None]
    assert attr(root, "langfuse.observation.level") == "ERROR"


def test_off_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    observability.langfuse_client.cache_clear()
    assert observability.langfuse_client() is None
