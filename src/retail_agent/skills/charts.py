"""Skill charts (off by default): text bar charts for the terminal. Config:
config/skills/charts.yaml.

Example of a capability added without changing the agent. Production would render a PNG
to GCS instead.
"""

from langchain.tools import tool

from retail_agent.skills import Deps

WIDTH = 30


@tool
def bar_chart(title: str, labels: list[str], values: list[float]) -> str:
    """Draw a text bar chart (description: config/skills/charts.yaml)."""
    pairs = list(zip(labels, values, strict=False))[:20]
    top = max((abs(v) for _, v in pairs), default=0) or 1
    pad = max((len(label) for label, _ in pairs), default=0)
    bars = [
        f"{label:<{pad}} {'█' * round(WIDTH * abs(v) / top)} {v:,.0f}"
        for label, v in pairs
    ]
    return "\n".join([title, *bars])


def make_tools(deps: Deps, skill: dict) -> list:
    return [bar_chart]
