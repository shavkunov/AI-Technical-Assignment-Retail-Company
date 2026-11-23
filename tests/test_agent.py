from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from conftest import PROMPTS, FakeToolModel
from retail_agent.agent import UserCtx, build_agent
from retail_agent.reports import ReportStore
from test_golden import TopicEmbeddings

T = "`bigquery-public-data.thelook_ecommerce`"


def call(name, **args):
    return AIMessage(
        "", tool_calls=[{"name": name, "args": args, "id": f"{name}-{id(args)}"}]
    )


def make(script, db, tmp_path, store=None, model=None):
    model = model or FakeToolModel(messages=iter(script))
    store = store or ReportStore(tmp_path / "app.sqlite")
    agent = build_agent(model, db, store, prefs_path=tmp_path / "prefs.json")
    return agent, store


def ask(agent, ctx, payload):
    config = {"configurable": {"thread_id": ctx.conversation_id}}
    if isinstance(payload, str):
        payload = {"messages": [{"role": "user", "content": payload}]}
    return agent.invoke(payload, config, context=ctx)


def tool_texts(result):
    return [m.content for m in result["messages"] if isinstance(m, ToolMessage)]


CEO = UserCtx(user_id="ceo", role="admin", brands=None, conversation_id="c1")
ALICE = UserCtx(
    user_id="alice", role="brand_exec", brands=["Calvin Klein"], conversation_id="c2"
)


def test_self_corrects_after_sql_error(local_db, tmp_path):
    script = [
        call("run_sql", sql=f"SELECT SUM(sale_pric) FROM {T}.order_items"),
        call(
            "run_sql",
            sql=f"SELECT ROUND(SUM(sale_price)) AS revenue FROM {T}.order_items",
        ),
        AIMessage("Revenue is large."),
    ]
    agent, _ = make(script, local_db, tmp_path)
    result = ask(agent, CEO, "total revenue?")
    first, second = tool_texts(result)
    assert first.startswith("ERROR") and "sale_pric" in first
    assert "revenue" in second and not second.startswith("ERROR")
    assert result["messages"][-1].content == "Revenue is large."


def test_empty_result_gets_hint(local_db, tmp_path):
    script = [
        call("run_sql", sql=f"SELECT id FROM {T}.products WHERE brand = 'Nope'"),
        AIMessage("none"),
    ]
    agent, _ = make(script, local_db, tmp_path)
    assert "0 rows" in tool_texts(ask(agent, CEO, "q"))[0]


def test_pii_never_reaches_model(local_db, tmp_path):
    script = [
        call("run_sql", sql=f"SELECT email FROM {T}.users"),
        call(
            "run_sql",
            sql=f"SELECT user_id, SUM(sale_price) AS spend FROM {T}.order_items "
            "GROUP BY user_id ORDER BY spend DESC LIMIT 3",
        ),
        AIMessage("Top customers listed."),
    ]
    agent, _ = make(script, local_db, tmp_path)
    blocked, top = tool_texts(ask(agent, CEO, "top customers with emails"))
    assert "PII" in blocked
    assert "C-" in top and "@" not in top


def test_analyst_sees_only_own_brands(local_db, tmp_path):
    script = [
        call("run_sql", sql=f"SELECT DISTINCT brand FROM {T}.products"),
        AIMessage("ok"),
    ]
    agent, _ = make(script, local_db, tmp_path)
    text = tool_texts(ask(agent, ALICE, "brands?"))[0]
    assert "Calvin Klein" in text and "Levi" not in text and "Nike" not in text


def test_run_sql_capped_per_turn(local_db, tmp_path):
    class CountingDB:
        calls = 0

        def run(self, sql):
            CountingDB.calls += 1
            raise RuntimeError("boom")

        def schema(self):
            return "products:\n  id INTEGER"

    script = [
        call("run_sql", sql=f"SELECT {i} FROM {T}.products") for i in range(6)
    ] + [AIMessage("gave up")]
    agent, _ = make(script, CountingDB(), tmp_path)
    result = ask(agent, CEO, "q")
    assert CountingDB.calls == 4
    assert result["messages"][-1].content == "gave up"


def _store_with_reports(tmp_path):
    store = ReportStore(tmp_path / "app.sqlite")
    store.save("alice", "old", "Client X review", "Client X is growing")
    store.save("alice", "old", "Churn", "no clients here")
    store.save("bob", "old", "Client X for Bob", "Client X")
    return store


def test_delete_waits_for_confirmation_and_cancel_keeps_reports(local_db, tmp_path):
    store = _store_with_reports(tmp_path)
    script = [
        call("delete_reports", contains="Client X"),
        AIMessage("Cancelled, nothing deleted."),
    ]
    agent, _ = make(script, local_db, tmp_path, store)
    paused = ask(agent, ALICE, "delete all reports mentioning Client X")
    request = paused["__interrupt__"][0].value
    assert request["action"] == "delete"
    assert [r["title"] for r in request["items"]] == ["Client X review"]
    assert len(store.find("alice")) == 2  # nothing deleted while waiting

    done = ask(
        agent,
        ALICE,
        Command(resume={"approved": False, "ids": [r["id"] for r in request["items"]]}),
    )
    assert "cancel" in tool_texts(done)[-1].lower()
    assert len(store.find("alice")) == 2


def test_delete_after_approval_deletes_only_previewed_own_reports(local_db, tmp_path):
    store = _store_with_reports(tmp_path)
    script = [call("delete_reports", contains="Client X"), AIMessage("Deleted.")]
    agent, _ = make(script, local_db, tmp_path, store)
    request = ask(agent, ALICE, "delete Client X reports")["__interrupt__"][0].value
    bob_id = store.find("bob")[0]["id"]
    # a forged extra id must be ignored
    ids = [r["id"] for r in request["items"]] + [bob_id]
    ask(agent, ALICE, Command(resume={"approved": True, "ids": ids}))
    assert [r["title"] for r in store.find("alice")] == ["Churn"]
    assert len(store.find("bob")) == 1


def test_delete_this_conversation(local_db, tmp_path):
    store = _store_with_reports(tmp_path)
    script = [
        call("save_report", title="Today", body="Q1 summary"),
        call("delete_reports", this_conversation=True),
        AIMessage("Deleted."),
    ]
    agent, _ = make(script, local_db, tmp_path, store)
    request = ask(agent, ALICE, "save then delete what we made today")["__interrupt__"][
        0
    ].value
    assert [r["title"] for r in request["items"]] == ["Today"]


def test_schema_golden_and_quote_rule_are_in_the_prompt_without_tool_calls(
    local_db, tmp_path
):
    # Saves 2 model calls per question vs. describe_data + search_golden tools
    # (free-tier quota).
    agent, _ = make([AIMessage("ok")], local_db, tmp_path)
    ask(agent, CEO, "What is our monthly revenue trend?")
    system = PROMPTS[-1][0].text
    assert "sale_price" in system  # table schema
    assert "FORMAT_DATE" in system  # SQL of the most similar golden trio
    assert '"Levi\'s"' in system  # BigQuery string quoting rule


def test_prompt_says_when_to_ask_and_to_state_assumptions(local_db, tmp_path):
    agent, _ = make([AIMessage("ok")], local_db, tmp_path)
    ask(agent, CEO, "Why are users underspending?")
    system = PROMPTS[-1][0].text
    assert "clarifying question" in system  # ask when a parameter is missing
    assert "state the assumption" in system  # otherwise default and say so
    # after a failed or empty query caused by a value not in the data: ask, not guess
    assert "which one they meant" in system


def test_golden_uses_embeddings_when_given(local_db, tmp_path):
    # "sales" shares no word with the revenue trio: only the embedding finds it
    model = FakeToolModel(messages=iter([AIMessage("ok")]))
    agent = build_agent(
        model,
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        tmp_path / "p.json",
        embeddings=TopicEmbeddings(),
    )
    ask(agent, CEO, "How are sales doing over time?")
    assert "FORMAT_DATE" in PROMPTS[-1][0].text


def test_broken_golden_file_is_skipped(local_db, tmp_path, config_copy):
    golden = tmp_path / "golden"
    golden.mkdir()
    (golden / "broken.yaml").write_text("sql: SELECT 1\n")  # no 'question' key
    (golden / "bad.yaml").write_text("question: [unclosed\n")  # invalid YAML
    (golden / "ok.yaml").write_text(
        "question: revenue by month\nsql: SELECT 42\nreport: fine\n"
    )
    (tmp_path / "persona.md").write_text("Be brief.")
    model = FakeToolModel(messages=iter([AIMessage("answer")]))
    agent = build_agent(
        model,
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        tmp_path / "p.json",
        config_dir=tmp_path,
    )
    assert ask(agent, CEO, "revenue?")["messages"][-1].content == "answer"
    assert "SELECT 42" in PROMPTS[-1][0].text


def test_tool_crash_becomes_error_message_and_next_turn_works(local_db, tmp_path):
    class BrokenStore(ReportStore):
        def find(self, *args, **kwargs):
            raise RuntimeError("disk gone")

    script = [call("list_reports"), AIMessage("first"), AIMessage("second")]
    agent, _ = make(script, local_db, tmp_path, BrokenStore(tmp_path / "b.sqlite"))
    first = ask(agent, CEO, "q1")
    assert tool_texts(first)[0].startswith("ERROR")
    assert ask(agent, CEO, "q2")["messages"][-1].content == "second"


def test_daily_quota_error_switches_to_fallback_without_retries(local_db, tmp_path):
    calls = []

    class QuotaExhausted(FakeToolModel):
        def _generate(self, *args, **kwargs):
            calls.append(1)
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED quotaId: "
                "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
            )

    fallback = FakeToolModel(messages=iter([AIMessage("from fallback")]))
    agent = build_agent(
        QuotaExhausted(messages=iter([])),
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        tmp_path / "p.json",
        fallback=fallback,
    )
    assert ask(agent, CEO, "q")["messages"][-1].content == "from fallback"
    # retrying a daily quota only burns time; it resets at midnight Pacific:
    # one guardrail call and one agent call, no retry of either
    assert len(calls) == 2


def test_unrelated_question_gets_no_golden_trios(local_db, tmp_path):
    agent, _ = make([AIMessage("ok")], local_db, tmp_path)
    ask(agent, CEO, "I prefer tables")
    assert (
        "SELECT"
        not in PROMPTS[-1][0]
        .text.split("Similar past analyses")[1]
        .split("Current user")[0]
    )


def test_tool_output_is_capped(tmp_path):
    import pandas as pd

    class BlobDB:
        def run(self, sql):
            return pd.DataFrame({"blob": ["ab " * 1_000_000], "n": [1]})

        def schema(self):
            return ""

    agent, _ = make(
        [call("run_sql", sql=f"SELECT 1 FROM {T}.products"), AIMessage("ok")],
        BlobDB(),
        tmp_path,
    )
    text = tool_texts(ask(agent, CEO, "q"))[0]
    assert len(text) < 10_000 and "truncated" in text


def test_read_report_returns_own_body_only(local_db, tmp_path):
    store = _store_with_reports(tmp_path)
    bob_id = store.find("bob")[0]["id"]
    alice_id = store.find("alice", text="Client X")[0]["id"]
    script = [
        call("read_report", report_id=alice_id),
        call("read_report", report_id=bob_id),
        AIMessage("ok"),
    ]
    agent, _ = make(script, local_db, tmp_path, store)
    own, other = tool_texts(ask(agent, ALICE, "open my reports"))
    assert "Client X is growing" in own and other == "No such report."


def test_broken_config_does_not_break_turns(local_db, tmp_path, config_copy):
    (tmp_path / "golden").mkdir()
    prefs = tmp_path / "p.json"
    prefs.write_text("{not json")  # corrupt preferences file; persona.md missing too
    script = [
        call("remember_preference", topic="email", note="send to jane@corp.com"),
        AIMessage("saved"),
        AIMessage("next"),
    ]
    agent = build_agent(
        FakeToolModel(messages=iter(script)),
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        prefs,
        config_dir=tmp_path,
    )
    assert ask(agent, CEO, "remember")["messages"][-1].content == "saved"
    assert ask(agent, CEO, "q2")["messages"][-1].content == "next"


def test_run_sql_keeps_executed_sql_and_job_stats_out_of_the_model_text(tmp_path):
    import pandas as pd

    class StatsDB:
        def run(self, sql):
            df = pd.DataFrame({"n": [1]})
            df.attrs.update(job_id="job-1", bytes_processed=1234)
            return df

        def schema(self):
            return ""

    agent, _ = make(
        [call("run_sql", sql=f"SELECT 1 AS n FROM {T}.orders"), AIMessage("ok")],
        StatsDB(),
        tmp_path,
    )
    [msg] = [m for m in ask(agent, CEO, "q")["messages"] if isinstance(m, ToolMessage)]
    assert msg.artifact["job_id"] == "job-1" and msg.artifact["bytes_processed"] == 1234
    assert "LIMIT" in msg.artifact["sql"]  # the guarded SQL that actually ran
    assert "job-1" not in msg.content


def test_rejected_query_is_kept_for_the_trace_and_odd_config_is_tolerated(
    local_db, tmp_path, config_copy
):
    (tmp_path / "golden").mkdir()
    (tmp_path / "persona.md").write_bytes("Be brief.".encode("utf-16"))  # not UTF-8
    prefs = tmp_path / "p.json"
    prefs.write_text("[]")  # valid JSON, wrong shape
    script = [call("run_sql", sql=f"SELECT email FROM {T}.users"), AIMessage("refused")]
    agent = build_agent(
        FakeToolModel(messages=iter(script)),
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        prefs,
        config_dir=tmp_path,
    )
    [msg] = [
        m for m in ask(agent, CEO, "emails")["messages"] if isinstance(m, ToolMessage)
    ]
    assert msg.artifact == {"sql": f"SELECT email FROM {T}.users", "rejected": True}


def test_empty_result_for_an_analyst_names_the_visible_brands(local_db, tmp_path):
    # The brand rewrite makes other brands' rows invisible; without saying so the model
    # guessed ("Levi's sells more").
    script = [
        call("run_sql", sql=f"SELECT brand FROM {T}.products WHERE brand = 'Nike'"),
        AIMessage("ok"),
    ]
    agent, _ = make(script, local_db, tmp_path)
    text = tool_texts(ask(agent, ALICE, "nike?"))[0]
    assert "0 rows" in text and "Calvin Klein" in text


def test_empty_result_over_visible_brands_has_no_brand_hint(local_db, tmp_path):
    # Live run: 0 rows for "Calvin Klein"/"Levi's" this month got the "other brands
    # are invisible" hint although both brands are visible; it misleads the model.
    sql = (
        f"SELECT brand FROM {T}.products "
        "WHERE brand IN ('Calvin Klein', \"Levi's\") AND retail_price < 0"
    )
    agent, _ = make([call("run_sql", sql=sql), AIMessage("ok")], local_db, tmp_path)
    text = tool_texts(ask(agent, ALICE, "this month?"))[0]
    assert "0 rows" in text and "invisible" not in text


def test_primary_is_tried_again_when_the_fallback_is_out_of_quota(local_db, tmp_path):
    # Live run: flash-lite had a transient failure while the fallback's daily quota was
    # gone -> the whole turn failed.
    calls = []

    class Flaky(FakeToolModel):
        def _generate(self, *args, **kwargs):
            calls.append(1)
            if len(calls) <= 4:  # the first round (1 call + 3 retries) all fail
                raise RuntimeError("503 UNAVAILABLE")
            return super()._generate(*args, **kwargs)

    class OutOfQuota(FakeToolModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED "
                "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
            )

    agent = build_agent(
        Flaky(messages=iter([AIMessage("answer")])),
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        tmp_path / "p.json",
        fallback=OutOfQuota(messages=iter([])),
    )
    assert ask(agent, CEO, "q")["messages"][-1].content == "answer"


def test_rate_limit_retries_wait_long_enough_for_a_per_minute_quota(
    local_db, tmp_path, monkeypatch
):
    # Live run: free-tier requests-per-minute 429s outlasted 1 s + 2 s of backoff and
    # failed whole turns.
    from langchain.agents.middleware import model_retry

    waited, calls = [], []
    monkeypatch.setattr(model_retry.time, "sleep", waited.append)

    class PerMinuteLimited(FakeToolModel):
        def _generate(self, *args, **kwargs):
            calls.append(1)
            if len(calls) <= 3:
                raise RuntimeError(
                    "429 RESOURCE_EXHAUSTED "
                    "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
                )
            return super()._generate(*args, **kwargs)

    agent = build_agent(
        PerMinuteLimited(messages=iter([AIMessage("answer")])),
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        tmp_path / "p.json",
    )
    assert ask(agent, CEO, "q")["messages"][-1].content == "answer"
    assert sum(waited) >= 15  # seconds of backoff before giving up on the model
