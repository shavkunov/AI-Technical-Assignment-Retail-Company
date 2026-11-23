import shutil

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from retail_agent.config import ROOT
from retail_agent.db import LocalDB
from retail_agent.guardrail import GUARDRAIL


@pytest.fixture(scope="session")
def local_db():
    return LocalDB()


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    """Model retries back off for up to 26 s; tests do not wait (a test can still record
    the delays)."""
    from langchain.agents.middleware import model_retry

    monkeypatch.setattr(model_retry.time, "sleep", lambda seconds: None)


@pytest.fixture
def config_copy(tmp_path):
    """A writable copy of the code-owned config (prompts + skills) without
    persona/golden/users."""
    shutil.copy(ROOT / "config" / "prompts.yaml", tmp_path / "prompts.yaml")
    shutil.copytree(ROOT / "config" / "skills", tmp_path / "skills")
    return tmp_path


PROMPTS = []  # every message list sent to a FakeToolModel, newest last
BOUND_TOOLS = []  # every tool list bound to a FakeToolModel, newest last


class FakeToolModel(GenericFakeChatModel):
    """Scripted chat model: returns the given AIMessages in order, records the tools and
    prompts it gets."""

    def bind_tools(self, tools, **kwargs):
        if kwargs.get("tool_choice") != "classify":  # the guardrail's own tool
            BOUND_TOOLS.append(tools)
        return self

    label: str = "ANALYTICS"  # guardrail tool call label; not taken from the script
    # prompts the input guardrail sent, newest last
    classified: list[str] = Field(default_factory=list)

    def _generate(self, messages, *args, **kwargs):
        if messages and messages[0].name == GUARDRAIL:
            self.classified.append(messages[0].text)
            call = {"name": "classify", "args": {"label": self.label}, "id": "g"}
            reply = ChatGeneration(message=AIMessage("", tool_calls=[call]))
            return ChatResult(generations=[reply])
        PROMPTS.append(messages)
        return super()._generate(messages, *args, **kwargs)
