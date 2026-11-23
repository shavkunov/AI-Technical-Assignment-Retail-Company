"""Skill preferences: per-user preference notes. Config:
config/skills/preferences.yaml."""

import json
import threading
from pathlib import Path

from langchain.tools import ToolRuntime, tool

from retail_agent.skills import Deps, UserCtx

MAX_NOTES = 20  # per user; the oldest note is dropped first
MAX_CHARS = 200  # per note


class Preferences:
    """One note per topic per user in one JSON file (production: LangGraph
    PostgresStore, namespace ("memory", user_id))."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()  # parallel tool calls run in threads

    def get(self, user_id: str) -> list[str]:
        return [f"{t}: {n}" for t, n in self._user(self._load(), user_id).items()]

    def add(self, user_id: str, topic: str, note: str) -> None:
        """Upsert the topic's note (moved to newest); an empty note forgets it."""
        with self.lock:
            prefs = self._load()
            notes = self._user(prefs, user_id)
            topic = topic.strip()
            notes.pop(topic, None)
            if note := note.strip()[:MAX_CHARS]:
                notes[topic] = note
            prefs[user_id] = dict(list(notes.items())[-MAX_NOTES:])
            self.path.write_text(json.dumps(prefs, indent=2), encoding="utf-8")

    @staticmethod
    def _user(prefs: dict, user_id: str) -> dict[str, str]:
        notes = prefs.get(user_id)
        return notes if isinstance(notes, dict) else {}  # old list format: ignored

    def _load(self) -> dict:
        try:
            prefs = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            # a damaged file must not break every turn
            return {}
        return prefs if isinstance(prefs, dict) else {}


def make_tools(deps: Deps, skill: dict) -> list:
    @tool
    def remember_preference(
        topic: str, note: str, runtime: ToolRuntime[UserCtx]
    ) -> str:
        """Remember a preference (description: config/skills/preferences.yaml)."""
        deps.prefs.add(runtime.context.user_id, topic, note)
        return "Preference saved." if note.strip() else "Preference forgotten."

    return [remember_preference]
