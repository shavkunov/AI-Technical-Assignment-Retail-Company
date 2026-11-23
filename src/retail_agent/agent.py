"""The agent: a LangGraph graph built by LangChain's create_agent.

What the model reads comes from config: config/prompts.yaml (rules + context template),
config/skills/*.yaml
(capabilities: instructions + tools), config/persona.md (tone) and config/golden/*.yaml
(analyst examples).
This file only wires them together with the input guardrail and the resilience and cost
middleware.
"""

import functools
from pathlib import Path
from typing import NotRequired

import yaml
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    before_agent,
    dynamic_prompt,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver

from retail_agent import golden
from retail_agent.config import DATASET, ROOT
from retail_agent.guardrail import ALLOWED, GUARDRAIL, LABELS, classify
from retail_agent.observability import is_user_message
from retail_agent.reports import ReportStore
from retail_agent.skills import Deps, UserCtx, load_skills
from retail_agent.skills.preferences import Preferences

__all__ = ["UserCtx", "build_agent"]

MAX_SQL_PER_TURN = 4
GUARDRAIL_HISTORY = 3  # earlier user messages the input guardrail sees


class GuardrailState(AgentState):
    guardrail: NotRequired[str]  # label of the last classified user message


def _worth_retrying(error: Exception) -> bool:
    """Retry transient model errors. A daily quota (Gemini free tier) resets at midnight
    Pacific:
    retrying only burns time, so go straight to the fallback model."""
    return getattr(error, "is_retryable", True) and "PerDay" not in str(error)


def build_agent(
    model: BaseChatModel,
    db,
    store: ReportStore,
    prefs_path: Path,
    config_dir: Path = ROOT / "config",
    fallback: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
):
    prompts = yaml.safe_load((config_dir / "prompts.yaml").read_text(encoding="utf-8"))
    system = prompts["system"].format(dataset=DATASET)
    context_keys = (
        "skills",
        "schema",
        "golden",
        "user_id",
        "role",
        "scope",
        "preferences",
        "persona",
    )
    # a placeholder typo fails here, not per question
    prompts["context"].format(**dict.fromkeys(context_keys, ""))
    guard = prompts["guardrail"]
    guard["prompt"].format(dataset="", history="", message="")
    refusals = {label: guard["refusals"][label] for label in LABELS if label != ALLOWED}
    prefs = Preferences(prefs_path)
    skills = load_skills(
        config_dir, Deps(db, store, prefs), dataset=DATASET, max_sql=MAX_SQL_PER_TURN
    )
    skills_text = "\n\n".join(
        f"Skill {s.name}:\n{s.instructions}" for s in skills if s.instructions
    )
    # BigQuery metadata calls once per process, not per model call
    schema = functools.cache(db.schema)

    golden_cache: dict = {}  # embeddings of trios and of the current question

    def similar_golden(question: str) -> str:
        """Past analyst trios most similar to the question (hybrid search, see
        golden.py). Files are re-read, so new trios apply at once."""
        trios = []
        for path in sorted((config_dir / "golden").glob("*.yaml")):
            try:
                trio = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError:
                continue  # a broken file edited by a non-developer must not break chat
            if isinstance(trio, dict) and isinstance(trio.get("question"), str):
                trios.append(trio)
        best = golden.similar(question, trios, embeddings, golden_cache)
        return (
            "\n---\n".join(yaml.safe_dump(t, sort_keys=False) for t in best) or "none"
        )

    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        ctx: UserCtx = request.runtime.context
        persona_file = config_dir / "persona.md"  # re-read: edits apply instantly
        persona = (
            persona_file.read_text(encoding="utf-8", errors="replace")
            if persona_file.exists()
            else prompts["default_persona"]
        )
        question = next(
            (m.text for m in reversed(request.messages) if is_user_message(m)), ""
        )
        # Schema and golden trios go into the prompt instead of tools: 2 fewer model
        # calls per question.
        context = prompts["context"].format(
            skills=skills_text,
            schema=schema(),
            golden=similar_golden(question),
            user_id=ctx.user_id,
            role=ctx.role,
            scope="all brands" if ctx.brands is None else ", ".join(ctx.brands),
            preferences="\n".join(f"- {p}" for p in prefs.get(ctx.user_id))
            or "- none yet",
            persona=persona,
        )
        return f"{system}\n{context}"

    @before_agent(state_schema=GuardrailState, can_jump_to=["end"])
    def input_guardrail(state: GuardrailState, runtime) -> dict | None:
        """Classify each new user message before the agent sees it: only ANALYTICS
        goes on. Resumes are not user messages."""
        *earlier, message = state["messages"]
        if not is_user_message(message):
            return None
        history = [m.text for m in earlier if is_user_message(m)][-GUARDRAIL_HISTORY:]
        prompt = guard["prompt"].format(
            dataset=DATASET,
            history="\n".join(f"- {h}" for h in history) or "none",
            message=message.text,
        )
        # same order as the agent's ModelFallbackMiddleware below
        models = [model, fallback, model] if fallback else [model]
        label = classify(models, prompt, guard["tool_description"], _worth_retrying)
        if label == ALLOWED:
            return {"guardrail": label}
        # the refused text is dropped: later turns (agent and guardrail) never see it
        forget = RemoveMessage(id=message.id)
        refusal = AIMessage(refusals[label], name=GUARDRAIL)
        return {"guardrail": label, "messages": [forget, refusal], "jump_to": "end"}

    middleware = [
        input_guardrail,
        system_prompt,
        # primary -> fallback -> primary, each with its retries (worst case 12 attempts
        # per model call): a short primary outage does not kill the turn
        # when the fallback is down too (live run: free-tier fallback quota gone during
        # a primary 5xx).
        *([ModelFallbackMiddleware(fallback, model)] if fallback else []),
        # 2 s, 6 s, 18 s: long enough to outlast a free-tier requests-per-minute window
        # (live run), short enough
        # that a real outage moves on to the fallback within ~30 s.
        ModelRetryMiddleware(
            max_retries=3,
            initial_delay=2.0,
            backoff_factor=3.0,
            max_delay=20.0,
            on_failure="error",
            retry_on=_worth_retrying,
        ),
        ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),
        ToolCallLimitMiddleware(
            tool_name="run_sql", run_limit=MAX_SQL_PER_TURN, exit_behavior="continue"
        ),
        # A crashing tool must still answer its tool call, or the checkpoint keeps a
        # dangling call.
        ToolErrorMiddleware(
            on_error=lambda e, _request: prompts["tool_error"].format(
                error=type(e).__name__
            )
        ),
    ]
    return create_agent(
        model,
        tools=[t for s in skills for t in s.tools],
        middleware=middleware,
        context_schema=UserCtx,
        checkpointer=InMemorySaver(),
    )
