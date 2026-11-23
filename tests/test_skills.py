"""Prompts and skills are configuration (config/prompts.yaml, config/skills/*.yaml), not
code."""

import yaml
from langchain_core.messages import AIMessage

from conftest import BOUND_TOOLS, PROMPTS, FakeToolModel
from retail_agent.agent import UserCtx, build_agent
from retail_agent.config import ROOT
from retail_agent.reports import ReportStore
from retail_agent.skills.charts import bar_chart

CEO = UserCtx(user_id="ceo", role="admin", brands=None, conversation_id="s1")


def run_once(local_db, tmp_path, config_dir=ROOT / "config"):
    agent = build_agent(
        FakeToolModel(messages=iter([AIMessage("ok")])),
        local_db,
        ReportStore(tmp_path / "a.sqlite"),
        tmp_path / "p.json",
        config_dir=config_dir,
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "monthly revenue?"}]},
        {"configurable": {"thread_id": "s1"}},
        context=CEO,
    )
    return {t.name: t.description for t in BOUND_TOOLS[-1]}, PROMPTS[-1][0].text


def edit_yaml(path, **changes):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_default_skills_give_tools_with_yaml_descriptions_and_instructions(
    local_db, tmp_path
):
    tools, prompt = run_once(local_db, tmp_path)
    assert set(tools) == {
        "run_sql",
        "save_report",
        "list_reports",
        "read_report",
        "delete_reports",
        "remember_preference",
    }
    sql_skill = yaml.safe_load(
        (ROOT / "config" / "skills" / "sql_analysis.yaml").read_text(encoding="utf-8")
    )
    assert tools["run_sql"] == sql_skill["tools"]["run_sql"]
    assert '"Levi\'s"' in prompt  # sql_analysis instructions
    assert "bar_chart" not in tools  # charts skill ships disabled


def test_prompt_text_comes_from_prompts_yaml(local_db, tmp_path, config_copy):
    edit_yaml(
        config_copy / "prompts.yaml",
        system="You are the CFO's analyst. Data: `{dataset}`.",
    )
    _, prompt = run_once(local_db, tmp_path, config_copy)
    assert prompt.startswith(
        "You are the CFO's analyst. Data: `bigquery-public-data.thelook_ecommerce`."
    )


def test_enabling_a_skill_in_yaml_adds_its_tools_and_instructions(
    local_db, tmp_path, config_copy
):
    edit_yaml(config_copy / "skills" / "charts.yaml", enabled=True)
    tools, prompt = run_once(local_db, tmp_path, config_copy)
    assert "bar_chart" in tools and "bar_chart" in prompt


def test_disabling_a_skill_removes_its_tools(local_db, tmp_path, config_copy):
    edit_yaml(config_copy / "skills" / "saved_reports.yaml", enabled=False)
    tools, _ = run_once(local_db, tmp_path, config_copy)
    assert not {"save_report", "delete_reports"} & set(tools) and "run_sql" in tools


def test_bar_chart_draws_scaled_bars():
    text = bar_chart.invoke(
        {"title": "Revenue", "labels": ["CK", "Levi's"], "values": [200.0, 100.0]}
    )
    ck, levis = text.splitlines()[1:3]
    assert ck.count("█") == 2 * levis.count("█") and "200" in ck


def test_a_tool_described_in_yaml_but_missing_in_the_module_fails_at_startup(
    local_db, tmp_path, config_copy
):
    import pytest

    path = config_copy / "skills" / "charts.yaml"
    edit_yaml(
        path,
        enabled=True,
        tools={"bar_chart": "x", "pie_chart": "typo or not implemented"},
    )
    with pytest.raises(ValueError, match="pie_chart"):
        run_once(local_db, tmp_path, config_copy)


def test_broken_prompt_templates_fail_at_startup(local_db, tmp_path, config_copy):
    import pytest

    edit_yaml(config_copy / "prompts.yaml", context="{schema} {no_such_placeholder}")
    # at build time, not at the first question
    with pytest.raises(KeyError, match="no_such_placeholder"):
        build_agent(
            FakeToolModel(messages=iter([])),
            local_db,
            ReportStore(tmp_path / "a.sqlite"),
            tmp_path / "p.json",
            config_dir=config_copy,
        )
