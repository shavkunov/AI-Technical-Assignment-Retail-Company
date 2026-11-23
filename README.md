# Retail Data Analysis Agent

CLI chat agent over BigQuery `thelook_ecommerce`: answers questions about sales, customers and products,
writes reports and manages saved reports. Design: [docs/](docs/).

## Setup

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (free key: <https://aistudio.google.com/apikey>).
By default the agent runs on a local synthetic copy of the data (`DATA_BACKEND=local`). For real BigQuery, set
`DATA_BACKEND=bigquery` and `GCP_PROJECT=...`, then run `gcloud auth application-default login`.

### Docker

```bash
docker compose run --build --rm agent --user ceo  # --build: image follows the code
```

This also starts Langfuse: tracing UI at <http://localhost:3000>, login `admin@retail.test` /
`langfuse-local`. Screenshots: [technical design §R7](docs/technical-design.md#r7-observability).

### uv

```bash
uv sync
uv run pytest                  # offline, no keys
uv run retail-agent --user ceo
```

### pip

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -e . pytest
retail-agent --user ceo
```

## Usage

`--user` picks a user from `config/users.yaml` (`ceo` sees everything, `alice` only her brands).
Ask in plain English, for example:

- `Top 5 customers by total spend`
- `Create a Q1 report with action items for Q2, and save it`
- `Delete all the reports we made in this conversation` (asks for `yes` first)

Commands: `/reports`, `/trace`, `/new`, `/exit`.
