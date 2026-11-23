"""One JSONL record per user turn: enough to replay what the agent saw, did and why it
failed.

Deep traces (every model and tool call) also go to Langfuse when LANGFUSE_PUBLIC_KEY is
set (docker compose runs it locally), and to LangSmith when LANGSMITH_TRACING=true.
"""

import functools
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from retail_agent.guardrail import GUARDRAIL

MAX_TEXT = 2000


def turn_record(
    trace_id: str,
    ctx,
    question: str,
    messages: list[BaseMessage],
    latency_ms: int,
    error: str | None,
) -> dict:
    # Messages of this turn = everything after the user's last message.
    last_human = max(
        (i for i, m in enumerate(messages) if is_user_message(m)), default=-1
    )
    turn = messages[last_human + 1 :]
    if messages and messages[-1].name == GUARDRAIL:
        turn = messages[-1:]  # refused: the user message was removed
    replies = [m for m in turn if isinstance(m, AIMessage)]
    ai = [m for m in replies if m.name != GUARDRAIL]  # a refusal is not a model call
    tools = [m for m in turn if isinstance(m, ToolMessage)]
    usage = [m.usage_metadata or {} for m in ai]
    return {
        "ts": datetime.now(UTC).isoformat(),
        "trace_id": trace_id,
        "user": ctx.user_id,
        "conversation_id": ctx.conversation_id,
        "question": question,
        "latency_ms": latency_ms,
        "error": error,
        "model_calls": len(ai),
        # Which models answered: a fallback model here means the primary one failed.
        "models": sorted({m.response_metadata.get("model_name") for m in ai} - {None}),
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "tool_calls": [
            {"name": c["name"], "args": c["args"]} for m in ai for c in m.tool_calls
        ],
        "tool_results": [
            {"name": m.name, "content": str(m.content)[:MAX_TEXT]} for m in tools
        ],
        # SQL that really ran after the guard rewrite, with BigQuery job id and bytes
        # (run_sql's artifact).
        "queries": [m.artifact for m in tools if m.name == "run_sql" and m.artifact],
        "sql_errors": sum(
            1
            for m in tools
            if m.name == "run_sql" and str(m.content).startswith("ERROR")
        ),
        "answer": replies[-1].text[:MAX_TEXT] if replies else None,
    }


def is_user_message(m: BaseMessage) -> bool:
    return isinstance(m, HumanMessage)


def write_trace(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


@functools.cache
def langfuse_client():
    """Langfuse client from LANGFUSE_* env vars, or None when tracing is off."""
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    return Langfuse()


@contextmanager
def langfuse_turn(ctx, question: str, trace_id: str):
    """One Langfuse trace per user turn. Yields `callbacks` for the agent config (the
    guardrail and every model and tool call nest under the turn)
    and `end(answer, record)`."""
    client = langfuse_client()
    if client is None:
        yield SimpleNamespace(callbacks=[], end=lambda answer, record: None)
        return

    def end(answer: str, record: dict) -> None:
        keys = ("guardrail", "sql_errors")
        root.update(
            output=answer,
            level="ERROR" if record["error"] else "DEFAULT",
            status_message=record["error"],
            metadata={k: record.get(k) for k in keys},
        )

    with (
        client.start_as_current_observation(
            name="turn", as_type="agent", input=question
        ) as root,
        propagate_attributes(
            user_id=ctx.user_id,
            session_id=ctx.conversation_id,
            trace_name="turn",
            metadata={"trace_id": trace_id, "role": ctx.role},
        ),
    ):
        yield SimpleNamespace(callbacks=[CallbackHandler()], end=end)
    client.flush()  # the CLI shows the next prompt only after the trace is sent
