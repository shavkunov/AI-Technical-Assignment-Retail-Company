"""Repo rules ruff cannot express. Ruff has no max-lines-per-file rule (pylint C0302 is
not ported)."""

from retail_agent.config import ROOT

MAX_LINES = 500


def test_no_python_file_over_max_lines():
    files = [*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]
    too_long = {
        str(p.relative_to(ROOT)): n
        for p in files
        if (n := len(p.read_text(encoding="utf-8").splitlines())) > MAX_LINES
    }
    assert files and not too_long, f"files over {MAX_LINES} lines: {too_long}"


def test_application_sql_lives_in_config_not_in_python():
    import re

    statement = re.compile(
        r"\bSELECT\b.*\bFROM\b|\bINSERT INTO\b|\bDELETE FROM\b|\bCREATE TABLE\b"
    )
    src = [
        p for p in (ROOT / "src").rglob("*.py") if p.name != "bq_client.py"
    ]  # client's file, verbatim
    found = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in src
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if statement.search(line) and not line.lstrip().startswith("#")  # examples
    ]
    assert not found, f"SQL belongs in config/sql.yaml: {found}"


def test_users_are_executives_not_analysts():
    """The users are non-technical executives (assignment). "Analyst" means the human
    experts who write golden trios, and the role goes into the prompt, so no user may
    have that role."""
    import yaml

    users = yaml.safe_load((ROOT / "config" / "users.yaml").read_text(encoding="utf-8"))
    assert {u["role"] for u in users.values()} == {"admin", "brand_exec"}


def test_compose_agent_starts_the_langfuse_worker():
    # Langfuse v4 web only queues events; the worker writes them where the UI reads.
    import yaml

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "langfuse-worker" in compose["services"]["agent"]["depends_on"]
