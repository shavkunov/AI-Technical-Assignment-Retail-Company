"""Skill sql_analysis: run guarded SQL. Config: config/skills/sql_analysis.yaml."""

from langchain.tools import ToolRuntime, tool

from retail_agent.pii import mask_frame
from retail_agent.skills import Deps, UserCtx
from retail_agent.sql_guard import SqlRejected, guard

MAX_ROWS_TO_MODEL = 50
# one huge cell (REPEAT, STRING_AGG) must not inflate every later model call
MAX_RESULT_CHARS = 8000


def make_tools(deps: Deps, skill: dict) -> list:
    messages = skill["messages"]

    @tool(response_format="content_and_artifact")
    def run_sql(sql: str, runtime: ToolRuntime[UserCtx]) -> tuple[str, dict]:
        """Run one read-only SELECT query (description:
        config/skills/sql_analysis.yaml)."""
        # The artifact (SQL that really ran, BigQuery job id and bytes) goes to the
        # trace, not to the model.
        try:
            safe_sql = guard(sql, runtime.context.brands)
        except SqlRejected as e:
            return f"ERROR (query rejected): {e}", {"sql": sql, "rejected": True}
        try:
            df = deps.db.run(safe_sql)
        # database errors go back to the model so it can fix the SQL
        except Exception as e:
            return f"ERROR (database): {str(e)[:800]}", {"sql": safe_sql}
        stats = {"sql": safe_sql, **df.attrs}
        if df.empty:
            brands = runtime.context.brands
            # ponytail: substring match; a query naming a visible and a hidden brand
            # skips the hint too. Parse the brand filter if that case shows up.
            scoped = (
                ""
                if brands is None or any(b.lower() in sql.lower() for b in brands)
                else " "
                + messages["empty_result_scoped"].format(brands=", ".join(brands))
            )
            return messages["empty_result"] + scoped, stats
        text = mask_frame(df).head(MAX_ROWS_TO_MODEL).to_csv(index=False)
        if len(df) > MAX_ROWS_TO_MODEL:
            text += "\n" + messages["rows_shown"].format(
                shown=MAX_ROWS_TO_MODEL, total=len(df)
            )
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "\n" + messages["truncated"]
        return text, stats

    return [run_sql]
