"""Result-side PII masking. Second line of defence after sql_guard (which blocks PII
columns)."""

import hashlib
import os

import pandas as pd

from retail_agent.config import PII_COLUMNS

ID_COLUMNS = {"user_id", "customer_id"}


def pseudonym(value) -> str:
    salt = os.environ.get("PII_SALT", "dev-salt")
    return "C-" + hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:6]


def mask_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in df.columns if c.lower() in PII_COLUMNS])
    for col in df.columns:
        if col.lower() in ID_COLUMNS:
            df[col] = df[col].map(pseudonym)
    return df
