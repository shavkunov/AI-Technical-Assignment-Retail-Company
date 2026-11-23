"""Settings from environment (.env). Config files live in ./config and are re-read on
use."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    llm_model: str
    llm_fallback_model: str
    embed_model: str  # golden trio search; empty = TF-IDF only
    data_backend: str  # "local" | "bigquery"
    gcp_project: str | None
    max_bytes_billed: int
    data_dir: Path
    config_dir: Path


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    return Settings(
        llm_model=env("LLM_MODEL", "google_genai:gemini-3.5-flash-lite"),
        llm_fallback_model=env("LLM_FALLBACK_MODEL", "google_genai:gemini-3.8-flash"),
        embed_model=env("EMBED_MODEL", "gemini-embedding-001"),
        data_backend=env("DATA_BACKEND", "local"),
        gcp_project=env("GCP_PROJECT") or None,
        max_bytes_billed=int(env("MAX_BYTES_BILLED", 2_000_000_000)),
        data_dir=data_dir,
        config_dir=ROOT / "config",
    )


DATASET = "bigquery-public-data.thelook_ecommerce"
TABLES = ("orders", "order_items", "products", "users")
# Never selectable, never shown. Pseudonymised: user_id (see pii.py).
PII_COLUMNS = frozenset(
    {
        "first_name",
        "last_name",
        "email",
        "street_address",
        "postal_code",
        "latitude",
        "longitude",
        "user_geom",
    }
)

# SQL the application runs itself (reports store, brand filter, local schema).
SQL = yaml.safe_load((ROOT / "config" / "sql.yaml").read_text(encoding="utf-8"))
