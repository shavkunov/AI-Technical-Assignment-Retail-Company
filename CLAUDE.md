# Project rules

OpsFleet take-home: CLI data-analysis agent over BigQuery `thelook_ecommerce`. The assignment is
[docs/assignment.md](docs/assignment.md); every change is checked against it
([docs/requirements-checklist.md](docs/requirements-checklist.md) maps each line to code, test and live proof).

## Skills (from USING_SKILLS.md): use all of them, while writing and as final checks

| Plugin | While writing | Final check |
|---|---|---|
| ponytail | `ponytail` (laziest working solution) | `ponytail-audit` (whole repo), `ponytail-review` (diff), `ponytail-debt` (ledger) |
| superpowers | `brainstorming`, `writing-plans`, `test-driven-development`, `systematic-debugging` | `requesting-code-review` (against the assignment), `verification-before-completion` |
| andrej-karpathy-skills | `karpathy-guidelines` (think first, simplicity, surgical changes, verifiable goals) | `karpathy-guidelines` review of the diff |
| caveman | `caveman` (terse replies) | `caveman-review` |
| code-simplifier | | `code-simplifier` agent |

Repeat the final-check round and fix the feedback until a round finds nothing actionable.

## Rules

- Do not over-complicate the base version. No new dependency for what a few lines can do.
- Prompts and capabilities are config: rules in `config/prompts.yaml`, each capability a skill
  (`config/skills/<name>.yaml` + `src/retail_agent/skills/<name>.py`). Instructions and hints the model reads
  (rules, skill instructions, tool descriptions, retry nudges) go in YAML; short status results produced by the
  logic itself ("Saved report #3", guard rejection reasons) stay next to that logic.
- TDD: a failing test first for every fix; `uv run pytest` (offline, no keys) must stay green.
- Lines at most 88 columns (Black / Ruff default, enforced with E501); no `.py` file over 500 lines
  and no SQL statements in Python (`tests/test_repo.py`; SQL lives in `config/sql.yaml`).
- pre-commit runs ruff lint + format, whitespace/YAML/TOML checks, a 500 KB file limit and the
  offline tests on every commit (`uv run pre-commit install` once). Do not bypass it.
- Default model `google_genai:gemini-3.5-flash-lite` (free tier); keep model calls per question low.
- `src/retail_agent/bq_client.py` is provided by the client: keep it verbatim.
- Git: commit to `master`, one commit per change. Stage explicit paths only, never `git add -A`: the folder
  can hold private documents that must never be committed (they are listed in `.git/info/exclude`).
- After behaviour changes, re-run `uv run python scripts/live_demo.py` (needs keys) to refresh docs/live-demo.md.
