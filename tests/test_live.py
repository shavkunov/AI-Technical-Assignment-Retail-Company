"""Real Gemini / BigQuery. Run with: uv run pytest -m live (skips what is not
configured)."""

import os

import pytest
from dotenv import load_dotenv

from retail_agent.config import ROOT, load_settings

load_dotenv(ROOT / ".env")  # before the skipif checks below
pytestmark = pytest.mark.live
T = "`bigquery-public-data.thelook_ecommerce`"


@pytest.mark.skipif(not os.environ.get("GCP_PROJECT"), reason="GCP_PROJECT not set")
def test_bigquery_guarded_query():
    from retail_agent.db import BigQueryDB
    from retail_agent.sql_guard import guard

    settings = load_settings()
    db = BigQueryDB(settings.gcp_project, settings.max_bytes_billed)
    df = db.run(
        guard(f"SELECT DISTINCT brand FROM {T}.products", brands=["Calvin Klein"])
    )
    assert list(df["brand"]) == ["Calvin Klein"]


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)
def test_gemini_answers_on_local_data(local_db, tmp_path):
    from retail_agent.agent import UserCtx, build_agent
    from retail_agent.cli import make_model, run_turn
    from retail_agent.reports import ReportStore

    settings = load_settings()
    agent = build_agent(
        make_model(settings.llm_model),
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        tmp_path / "p.json",
        fallback=make_model(settings.llm_fallback_model),
    )
    ctx = UserCtx("ceo", "admin", None, "live")
    answer = run_turn(
        agent,
        ctx,
        "How many orders do we have in total?",
        lambda r: False,
        tmp_path / "t.jsonl",
    )
    assert "went wrong" not in answer and any(ch.isdigit() for ch in answer)
