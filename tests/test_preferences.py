"""User memory: one note per topic per user, limited, put into the prompt."""

from retail_agent.skills.preferences import MAX_CHARS, MAX_NOTES, Preferences


def test_new_note_on_same_topic_replaces_old_and_users_are_separate(tmp_path):
    prefs = Preferences(tmp_path / "p.json")
    prefs.add("a", "format", "prefers tables")
    prefs.add("b", "format", "prefers bullet points")
    prefs.add("a", "format", "prefers bullet points")
    prefs.add("a", "depth", "short answers")
    assert prefs.get("a") == ["format: prefers bullet points", "depth: short answers"]
    assert prefs.get("b") == ["format: prefers bullet points"]


def test_empty_note_forgets_topic(tmp_path):
    prefs = Preferences(tmp_path / "p.json")
    prefs.add("a", "charts", "likes charts")
    prefs.add("a", "charts", "")
    assert prefs.get("a") == []


def test_limits_drop_oldest_note_and_cap_length(tmp_path):
    prefs = Preferences(tmp_path / "p.json")
    for i in range(MAX_NOTES + 1):
        prefs.add("a", f"t{i}", "x" * (MAX_CHARS + 50))
    notes = prefs.get("a")
    assert len(notes) == MAX_NOTES and notes[0].startswith("t1: ")
    assert all(len(n.split(": ", 1)[1]) == MAX_CHARS for n in notes)


def test_old_list_format_is_ignored(tmp_path):
    path = tmp_path / "p.json"
    path.write_text('{"a": ["prefers tables"]}')
    prefs = Preferences(path)
    assert prefs.get("a") == []
    prefs.add("a", "format", "prefers tables")
    assert prefs.get("a") == ["format: prefers tables"]
