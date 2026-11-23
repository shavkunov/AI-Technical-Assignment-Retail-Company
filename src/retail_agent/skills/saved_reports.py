"""Skill saved_reports: the reports library with a confirmed delete. Config:
config/skills/saved_reports.yaml."""

import json

from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

from retail_agent.skills import Deps, UserCtx


def make_tools(deps: Deps, skill: dict) -> list:
    store = deps.store

    @tool
    def save_report(title: str, body: str, runtime: ToolRuntime[UserCtx]) -> str:
        """Save a report (description: config/skills/saved_reports.yaml)."""
        ctx = runtime.context
        report_id = store.save(ctx.user_id, ctx.conversation_id, title, body)
        return f"Saved report #{report_id} '{title}'."

    @tool
    def list_reports(runtime: ToolRuntime[UserCtx], contains: str = "") -> str:
        """List reports (description: config/skills/saved_reports.yaml)."""
        rows = store.find(runtime.context.user_id, text=contains or None)
        return json.dumps(rows) if rows else "No saved reports."

    @tool
    def read_report(report_id: int, runtime: ToolRuntime[UserCtx]) -> str:
        """Read a report (description: config/skills/saved_reports.yaml)."""
        report = store.get(runtime.context.user_id, report_id)
        return (
            f"# {report['title']}\n\n{report['body']}" if report else "No such report."
        )

    @tool
    def delete_reports(
        runtime: ToolRuntime[UserCtx],
        contains: str = "",
        this_conversation: bool = False,
    ) -> str:
        """Delete reports after confirmation (description:
        config/skills/saved_reports.yaml)."""
        ctx = runtime.context
        matches = store.find(
            ctx.user_id,
            text=contains or None,
            conversation_id=ctx.conversation_id if this_conversation else None,
        )
        if not matches:
            return "No matching reports, nothing to delete."
        # The graph pauses for the user. {"action", "items"} is the shape every
        # confirmed action uses (cli.py).
        answer = interrupt({"action": "delete", "items": matches})
        if not answer.get("approved"):
            return "User cancelled. Nothing was deleted."
        # Only ids that were shown AND still match AND are owned (store.delete re-checks
        # the owner).
        ids = [i for i in answer.get("ids", []) if i in {r["id"] for r in matches}]
        return f"Deleted {store.delete(ctx.user_id, ids)} report(s)."

    return [save_report, list_reports, read_report, delete_reports]
