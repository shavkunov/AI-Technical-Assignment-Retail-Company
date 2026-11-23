"""Pluggable skills: what the agent can do is configuration, not agent code.

A skill is `config/skills/<name>.yaml` (enabled flag, instructions for the model, tool
descriptions, messages)
plus a Python module with `make_tools(deps, skill) -> list[BaseTool]`. Adding a
capability (charts, e-mailing
a report, web search) = one module + one YAML file; `agent.py` does not change.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class UserCtx:
    """Who is asking. Comes from the login (SSO in production), never from chat text."""

    user_id: str
    role: str
    brands: list[str] | None  # None = all brands (admin)
    conversation_id: str


@dataclass(frozen=True)
class Deps:
    """What skill tools may use."""

    db: object  # LocalDB | BigQueryDB
    store: object  # ReportStore
    prefs: object  # skills.preferences.Preferences


@dataclass(frozen=True)
class Skill:
    name: str
    instructions: str
    tools: list[BaseTool]


def load_skills(config_dir: Path, deps: Deps, **placeholders) -> list[Skill]:
    """Enabled skills from config/skills/*.yaml. A broken skill file fails at startup,
    loudly."""
    skills = []
    for path in sorted((config_dir / "skills").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not spec.get("enabled", True):
            continue
        described = spec.get("tools") or {}
        tools = [
            t
            for t in importlib.import_module(spec["module"]).make_tools(deps, spec)
            if t.name in described
        ]
        if missing := set(described) - {t.name for t in tools}:
            raise ValueError(
                f"{path.name}: tools not provided by "
                f"{spec['module']}: {', '.join(sorted(missing))}"
            )
        for t in tools:
            # the model reads the YAML text, not the docstring
            t.description = described[t.name]
        instructions = (spec.get("instructions") or "").format(**placeholders).strip()
        skills.append(Skill(spec["name"], instructions, tools))
    return skills
