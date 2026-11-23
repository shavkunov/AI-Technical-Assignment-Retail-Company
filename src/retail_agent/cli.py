"""Terminal chat. Usage: retail-agent --user alice"""

import argparse
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import yaml
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from retail_agent.agent import UserCtx, build_agent
from retail_agent.config import ROOT, load_settings
from retail_agent.db import make_db
from retail_agent.observability import langfuse_turn, turn_record, write_trace
from retail_agent.reports import ReportStore

console = Console()
HELP = "Commands: /reports  /trace (last turn)  /new (new conversation)  /exit"


def make_model(name: str) -> BaseChatModel | None:
    """Retries belong to ModelRetryMiddleware alone. The Gemini SDK would add 6 silent
    retries and no timeout.
    An empty name (e.g. LLM_FALLBACK_MODEL=) means no model."""
    if not name:
        return None
    if name.startswith("google_genai:"):
        # ponytail: free tier is 15 requests per sliding minute per model; a burst of 5
        # plus 10/min refill never exceeds it in any 60 s. Raise both on a paid key.
        limiter = InMemoryRateLimiter(requests_per_second=10 / 60, max_bucket_size=5)
        limiter.available_tokens = 5  # starts empty: the first call would wait 6 s
        # 1 = one attempt (0 means "SDK default")
        return init_chat_model(name, max_retries=1, timeout=60, rate_limiter=limiter)
    return init_chat_model(name)


def make_embeddings(name: str) -> Embeddings | None:
    """Gemini embeddings for the golden trio search. No key (e.g. an Ollama run) or an
    empty EMBED_MODEL means TF-IDF only."""
    if not name or not os.environ.get("GOOGLE_API_KEY"):
        return None
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=name)


def _invoke(
    agent, payload, config, ctx: UserCtx, confirm: Callable[[dict], bool]
) -> dict:
    result = agent.invoke(payload, config, context=ctx)
    # action(s) waiting for the user's yes/no, one answer each
    while "__interrupt__" in result:
        resume = {}
        for pending in result["__interrupt__"]:
            # request shape for every confirmed action: {"action": "delete", "items":
            # [{"id": ..., ...}]}
            try:
                ids = [item["id"] for item in pending.value["items"]]
                resume[pending.id] = {"approved": confirm(pending.value), "ids": ids}
            except Exception:
                # a malformed request or a broken prompt is still answered, as a cancel
                resume[pending.id] = {"approved": False, "ids": []}
        result = agent.invoke(Command(resume=resume), config, context=ctx)
    return result


def run_turn(
    agent, ctx: UserCtx, text: str, confirm: Callable[[dict], bool], trace_path: Path
) -> str:
    """One user turn. Never raises: failures become a friendly message plus a trace
    record."""
    trace_id, started, error = uuid.uuid4().hex[:12], time.monotonic(), None
    with langfuse_turn(ctx, text, trace_id) as turn:
        config = {
            "configurable": {"thread_id": ctx.conversation_id},
            "callbacks": turn.callbacks,
        }
        try:
            result = _invoke(
                agent,
                {"messages": [{"role": "user", "content": text}]},
                config,
                ctx,
                confirm,
            )
            answer = result["messages"][-1].text
        except Exception as e:
            # ponytail: one catch-all per turn keeps the UI alive; split per error type
            # when users need specific hints.
            error = f"{type(e).__name__}: {e}"
            answer = (
                "Sorry, something went wrong on my side "
                f"({type(e).__name__}). Please try again. [trace "
                f"{trace_id}]"
            )
        finally:
            try:
                values = agent.get_state(config).values
            except Exception:
                values = {}
            messages = values.get("messages", [])
            latency = int((time.monotonic() - started) * 1000)
            record = turn_record(trace_id, ctx, text, messages, latency, error)
            # input guardrail label; a failed turn may hold the previous turn's label
            record["guardrail"] = values.get("guardrail") if error is None else None
            write_trace(trace_path, record)
            turn.end(answer, record)
    return answer


def confirm_in_terminal(request: dict) -> bool:
    """Show exactly what an action will touch and ask for a typed yes. Works for any
    {"action", "items"}."""
    action, items = request["action"], request["items"]
    columns = list(items[0]) if items else ["id"]
    table = Table(
        *columns, title=f"About to {action.upper()} {len(items)} item(s)", style="red"
    )
    for item in items:
        # values are data, not markup
        table.add_row(*(escape(str(item.get(c, ""))) for c in columns))
    console.print(table)
    try:
        return (
            console.input(
                f"[bold red]Type 'yes' to {action}, anything else cancels: [/]"
            )
            .strip()
            .lower()
            == "yes"
        )
    except (EOFError, KeyboardInterrupt):
        return False


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Retail data analysis agent")
    parser.add_argument("--user", default="ceo", help="user id from config/users.yaml")
    args = parser.parse_args()

    settings = load_settings()
    users = yaml.safe_load(
        (settings.config_dir / "users.yaml").read_text(encoding="utf-8")
    )
    if args.user not in users:
        raise SystemExit(f"Unknown user '{args.user}'. Known: {', '.join(users)}")
    profile = users[args.user]
    brands = None if profile["role"] == "admin" else profile.get("brands", [])

    # empty fallback = none
    models = [m for m in (settings.llm_model, settings.llm_fallback_model) if m]
    if any(m.startswith("google_genai:") for m in models) and not os.environ.get(
        "GOOGLE_API_KEY"
    ):
        raise SystemExit(
            "GOOGLE_API_KEY is not set. Put it in .env (see README), "
            "or run offline with LLM_MODEL=ollama:<model> and "
            "an empty LLM_FALLBACK_MODEL."
        )
    store = ReportStore(settings.data_dir / "app.sqlite")
    agent = build_agent(
        make_model(settings.llm_model),
        make_db(settings),
        store,
        prefs_path=settings.data_dir / "prefs.json",
        config_dir=settings.config_dir,
        fallback=make_model(settings.llm_fallback_model),
        embeddings=make_embeddings(settings.embed_model),
    )
    trace_path = ROOT / "logs" / "traces.jsonl"
    ctx = UserCtx(args.user, profile["role"], brands, uuid.uuid4().hex[:8])
    console.print(
        "[bold]Retail analyst[/] - user "
        f"[cyan]{ctx.user_id}[/] ({settings.data_backend} "
        f"data). {HELP}"
    )

    while True:
        try:
            text = console.input("\n[bold green]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/exit":
            break
        if text == "/new":
            ctx = UserCtx(ctx.user_id, ctx.role, ctx.brands, uuid.uuid4().hex[:8])
            console.print("New conversation.")
        elif text == "/reports":
            console.print_json(json.dumps(store.find(ctx.user_id)))
        elif text == "/trace":
            lines = (
                trace_path.read_text(encoding="utf-8").splitlines()
                if trace_path.exists()
                else []
            )
            console.print_json(lines[-1] if lines else "{}")
        else:
            console.print("[dim]thinking... (Ctrl+C cancels this question)[/]")
            try:
                answer = run_turn(agent, ctx, text, confirm_in_terminal, trace_path)
            except KeyboardInterrupt:
                # the stopped turn may leave a half-done step: start a new conversation
                ctx = UserCtx(ctx.user_id, ctx.role, ctx.brands, uuid.uuid4().hex[:8])
                answer = "Cancelled. Starting a new conversation."
            console.print(Markdown(answer))


if __name__ == "__main__":
    main()
