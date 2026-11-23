import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from conftest import FakeToolModel
from retail_agent.agent import UserCtx, build_agent
from retail_agent.cli import main, make_embeddings, make_model, run_turn
from retail_agent.reports import ReportStore

ALICE = UserCtx(
    user_id="alice", role="brand_exec", brands=["Calvin Klein"], conversation_id="c1"
)


def read_traces(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_turn_survives_exception(tmp_path):
    class DownAgent:
        def invoke(self, *a, **k):
            raise RuntimeError("503 model overloaded")

        def get_state(self, config):
            return SimpleNamespace(values={"messages": []})

    answer = run_turn(
        DownAgent(),
        ALICE,
        "revenue?",
        confirm=lambda reports: True,
        trace_path=tmp_path / "t.jsonl",
    )
    assert "went wrong" in answer
    [record] = read_traces(tmp_path / "t.jsonl")
    assert "503 model overloaded" in record["error"] and record["user"] == "alice"


def test_turn_runs_confirmation_and_writes_trace(local_db, tmp_path):
    store = ReportStore(tmp_path / "app.sqlite")
    store.save("alice", "old", "Client X review", "Client X")
    script = [
        AIMessage(
            "",
            tool_calls=[
                {"name": "delete_reports", "args": {"contains": "Client X"}, "id": "d1"}
            ],
        ),
        AIMessage("Okay, kept them."),
    ]
    agent = build_agent(
        FakeToolModel(messages=iter(script)),
        local_db,
        store,
        prefs_path=tmp_path / "p.json",
    )
    shown = []
    answer = run_turn(
        agent,
        ALICE,
        "delete Client X reports",
        confirm=lambda r: shown.append(r) or False,
        trace_path=tmp_path / "t.jsonl",
    )

    assert shown[0]["action"] == "delete" and [
        r["title"] for r in shown[0]["items"]
    ] == ["Client X review"]
    assert len(store.find("alice")) == 1
    assert answer == "Okay, kept them."
    [record] = read_traces(tmp_path / "t.jsonl")
    assert record["error"] is None and record["latency_ms"] >= 0
    assert [c["name"] for c in record["tool_calls"]] == ["delete_reports"]
    assert record["question"] == "delete Client X reports"


def test_parallel_delete_calls_each_confirmed(local_db, tmp_path):
    store = ReportStore(tmp_path / "app.sqlite")
    store.save("alice", "old", "aa report", "x")
    store.save("alice", "old", "bb report", "x")
    two_calls = AIMessage(
        "",
        tool_calls=[
            {"name": "delete_reports", "args": {"contains": "aa"}, "id": "d1"},
            {"name": "delete_reports", "args": {"contains": "bb"}, "id": "d2"},
        ],
    )
    agent = build_agent(
        FakeToolModel(messages=iter([two_calls, AIMessage("done")])),
        local_db,
        store,
        prefs_path=tmp_path / "p.json",
    )
    asked = []
    answer = run_turn(
        agent,
        ALICE,
        "delete aa and bb",
        lambda r: asked.append(r) or True,
        tmp_path / "t.jsonl",
    )
    assert answer == "done" and len(asked) == 2
    assert store.find("alice") == []


def test_gemini_model_has_no_sdk_retries_and_a_timeout(monkeypatch):
    # SDK default is 6 silent retries and no timeout: with ModelRetryMiddleware on top
    # one question
    # burned the free daily quota and a 429 hung the turn instead of switching to the
    # fallback model.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    model = make_model("google_genai:gemini-3.5-flash-lite")
    # 1 = single attempt in the Google SDK (0 means "SDK default")
    assert model.max_retries == 1
    assert model.timeout == 60


def test_gemini_model_paces_calls_to_free_tier_quota(monkeypatch):
    # live run: 5 questions = 15 model calls in ~40 s, 429 "15 requests per minute".
    # A full bucket keeps the first calls instant; each model gets its own quota.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    primary = make_model("google_genai:gemini-3.5-flash-lite")
    fallback = make_model("google_genai:gemini-3.8-flash")
    limiter = primary.rate_limiter
    # any 60 s window: the full bucket plus a minute of refill must stay within 15
    assert limiter.max_bucket_size + limiter.requests_per_second * 60 <= 15
    assert limiter.available_tokens == limiter.max_bucket_size
    assert fallback.rate_limiter is not limiter


def test_make_embeddings_needs_key_and_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    assert make_embeddings("gemini-embedding-001") is None  # keyless / Ollama run
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert make_embeddings("") is None  # EMBED_MODEL= disables
    assert make_embeddings("gemini-embedding-001").model.endswith(
        "gemini-embedding-001"
    )


def test_missing_key_for_gemini_fallback_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "ollama:qwen3:8b")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "google_genai:gemini-3.8-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setattr("sys.argv", ["retail-agent"])
    with pytest.raises(SystemExit, match="GOOGLE_API_KEY"):
        main()


def test_empty_fallback_means_no_fallback(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "")
    from retail_agent.config import load_settings

    assert load_settings().llm_fallback_model == ""


def test_confirm_shows_markup_titles_literally(monkeypatch):
    from retail_agent import cli

    monkeypatch.setattr(cli.console, "input", lambda prompt: "no")
    request = {
        "action": "delete",
        "items": [{"id": 1, "title": "Q3 [/red] summary", "created_at": "now"}],
    }
    assert cli.confirm_in_terminal(request) is False


def test_failing_confirmation_cancels_and_conversation_continues(local_db, tmp_path):
    store = ReportStore(tmp_path / "app.sqlite")
    store.save("alice", "c1", "Q3", "x")
    script = [
        AIMessage(
            "",
            tool_calls=[
                {"name": "delete_reports", "args": {"contains": "Q3"}, "id": "d1"}
            ],
        ),
        AIMessage("Cancelled."),
        AIMessage("next answer"),
    ]
    agent = build_agent(
        FakeToolModel(messages=iter(script)), local_db, store, tmp_path / "p.json"
    )

    def broken_confirm(reports):
        raise RuntimeError("terminal gone")

    run_turn(agent, ALICE, "delete Q3", broken_confirm, tmp_path / "t.jsonl")
    assert len(store.find("alice")) == 1
    assert (
        run_turn(agent, ALICE, "hi", lambda r: False, tmp_path / "t.jsonl")
        == "next answer"
    )


def test_trace_records_models_and_query_stats(local_db, tmp_path):
    script = [
        AIMessage(
            "",
            tool_calls=[
                {
                    "name": "run_sql",
                    "args": {"sql": "SELECT COUNT(*) AS n FROM orders"},
                    "id": "s1",
                }
            ],
            response_metadata={"model_name": "fallback-model"},
        ),
        AIMessage("done", response_metadata={"model_name": "fallback-model"}),
    ]
    agent = build_agent(
        FakeToolModel(messages=iter(script)),
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        tmp_path / "p.json",
    )
    run_turn(agent, ALICE, "count", lambda r: False, tmp_path / "t.jsonl")
    [record] = read_traces(tmp_path / "t.jsonl")
    assert record["models"] == ["fallback-model"]
    # rewritten, brand-scoped SQL that actually ran
    assert "brand IN" in record["queries"][0]["sql"]


def test_any_confirmed_action_uses_the_same_prompt(monkeypatch):
    # A future skill (e.g. e-mailing a report) pauses with {"action", "items"}; the CLI
    # needs no change.
    from retail_agent import cli

    printed = []
    monkeypatch.setattr(cli.console, "print", lambda *a, **k: printed.append(a))
    monkeypatch.setattr(
        cli.console, "input", lambda prompt: printed.append(prompt) or "yes"
    )
    request = {"action": "email", "items": [{"id": 7, "to": "cfo", "report": "Q1"}]}
    assert cli.confirm_in_terminal(request) is True
    assert "EMAIL" in str(printed[0][0].title) and "email" in printed[1]


def test_malformed_confirmation_request_is_answered_as_cancel(tmp_path):
    resumed = []

    class PausingAgent:
        def invoke(self, payload, config, context):
            if isinstance(payload, Command):
                resumed.append(payload.resume)
                return {"messages": [AIMessage("cancelled")]}
            return {
                "__interrupt__": [SimpleNamespace(id="i1", value={"unexpected": True})]
            }

        def get_state(self, config):
            return SimpleNamespace(values={"messages": []})

    answer = run_turn(
        PausingAgent(), ALICE, "do it", lambda r: True, tmp_path / "t.jsonl"
    )
    assert answer == "cancelled" and resumed == [{"i1": {"approved": False, "ids": []}}]
