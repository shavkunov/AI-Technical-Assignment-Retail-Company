"""Deterministic SQL firewall. The LLM writes SQL; this decides what may run.

Prompt injection cannot get past it: only one read-only query over the four dataset
tables,
no PII columns, a row LIMIT, and (for non-admin users) every table swapped for a
subquery
that keeps only rows tied to the user's brands.
"""

import re

import sqlglot
from sqlglot import exp

from retail_agent.config import DATASET, PII_COLUMNS, SQL, TABLES

CATALOG, SCHEMA = DATASET.split(".")
FORBIDDEN = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
)


class SqlRejected(ValueError):
    """Message is shown to the model so it can fix the query."""


def guard(sql: str, brands: list[str] | None, limit: int = 200) -> str:
    """Return safe BigQuery SQL or raise SqlRejected. brands=None means admin (no row
    filter)."""
    try:
        try:
            parsed = sqlglot.parse(sql, read="bigquery")
        # models often write ANSI 'Levi''s'; BigQuery wants 'Levi\'s' (same length)
        except sqlglot.errors.SqlglotError:
            parsed = sqlglot.parse(
                re.sub(r"(?<=\w)''(?=\w)", r"\\'", sql), read="bigquery"
            )
        statements = [s for s in parsed if s is not None]
    # Parse/TokenError; str(e) has ANSI codes, send plain text to the model
    except sqlglot.errors.SqlglotError as e:
        err = (getattr(e, "errors", None) or [{}])[0]
        where = f" (line {err.get('line')}, col {err.get('col')})" if err else ""
        raise SqlRejected(
            f"SQL syntax error: {err.get('description', e)}{where}"
        ) from e
    if len(statements) != 1:
        raise SqlRejected("Exactly one SQL statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, exp.Query) or tree.find(*FORBIDDEN):
        raise SqlRejected("Only read-only SELECT queries are allowed.")

    # names are case-insensitive
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    # else `WITH products AS (SELECT * FROM products)` skips the brand filter
    if ctes & set(TABLES):
        raise SqlRejected(
            f"CTE names must not shadow table names ({', '.join(TABLES)})."
        )
    tables = [
        t for t in tree.find_all(exp.Table) if not (t.name.lower() in ctes and not t.db)
    ]
    for t in tables:
        if (
            t.name not in TABLES
            or t.db not in ("", SCHEMA)
            or t.catalog not in ("", CATALOG)
        ):
            raise SqlRejected(
                f"Unknown table {t.sql('bigquery')}. Allowed tables: "
                f"{', '.join(TABLES)}."
            )

    # BigQuery: case-insensitive
    user_aliases = {t.alias_or_name.lower() for t in tables if t.name == "users"}
    for col in tree.find_all(exp.Column):
        if col.name.lower() in PII_COLUMNS:
            raise SqlRejected(
                f"Column {col.name} is PII and cannot be queried. "
                "Use aggregates/demographics."
            )
        # SELECT u FROM users u -> whole row as STRUCT
        if not col.table and col.name.lower() in user_aliases:
            raise SqlRejected(
                "Selecting the whole users row is not allowed (PII). Name the columns."
            )
    if user_aliases and any(
        _is_row_star(s, user_aliases) for s in tree.find_all(exp.Star)
    ):
        raise SqlRejected("SELECT * over users is not allowed (PII). Name the columns.")
    # the brand rewrite drops them
    if any(t.args.get("version") for t in tables) or tree.find(exp.TableSample):
        raise SqlRejected("FOR SYSTEM_TIME AS OF and TABLESAMPLE are not supported.")

    for t in tables:
        t.set("catalog", exp.to_identifier(CATALOG, quoted=True))
        t.set("db", exp.to_identifier(SCHEMA))
        if brands is not None:
            t.replace(_brand_filter(t.name, brands).subquery(t.alias_or_name))

    existing = tree.args.get("limit")
    if existing is None or not (
        existing.expression.is_int and int(existing.expression.name) <= limit
    ):
        tree = tree.limit(limit)
    return tree.sql(dialect="bigquery")


def _is_row_star(star: exp.Star, user_aliases: set[str]) -> bool:
    parent = star.parent
    # u.* anywhere (STRUCT(u.*)), or any x.* in a SELECT list
    if isinstance(parent, exp.Column):
        return parent.table.lower() in user_aliases or isinstance(
            parent.parent, exp.Select
        )
    return isinstance(parent, exp.Select)  # SELECT *, not COUNT(*)


def _brand_filter(table: str, brands: list[str]) -> exp.Query:
    """The table as a subquery limited to the brands (config/sql.yaml: brand_filter)."""
    in_list = ", ".join(exp.Literal.string(b).sql("bigquery") for b in brands) or "NULL"
    sql = SQL["brand_filter"][table].format(dataset=DATASET, brands=in_list)
    return sqlglot.parse_one(sql, read="bigquery")
