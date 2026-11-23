"""Input guardrail: one model call forced to call the `classify` tool with a label.

The tool's `label` argument is an enum of LABELS and tool_choice forces the call, so the
label is read as is. The prompt, tool description and refusal texts live in
config/prompts.yaml (`guardrail:`).
"""

import time
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

ALLOWED = "ANALYTICS"
LABELS = (ALLOWED, "OFF_TOPIC", "MALICIOUS")
GUARDRAIL = "guardrail"  # name of the classifier's message and of a refusal
RETRY_DELAYS = (2.0, 6.0, 18.0)  # same policy as the agent's ModelRetryMiddleware
TOOL = "classify"


def classify(
    models: list[BaseChatModel],
    prompt: str,
    description: str,
    worth_retrying: Callable[[Exception], bool],
) -> str:
    """Label of the message in `prompt`. Each model is retried after RETRY_DELAYS, then
    the next model is tried; raises the last error when every model fails."""
    tool = {
        "name": TOOL,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(LABELS)}},
            "required": ["label"],
        },
    }
    error: Exception | None = None
    for model in models:
        model = model.bind_tools([tool], tool_choice=TOOL)
        for delay in (*RETRY_DELAYS, None):
            try:
                reply = model.invoke([HumanMessage(prompt, name=GUARDRAIL)])
            except Exception as e:
                error = e
                if delay is None or not worth_retrying(e):
                    break
                time.sleep(delay)
                continue
            return reply.tool_calls[0]["args"]["label"]
    raise error or ValueError("no model given")
