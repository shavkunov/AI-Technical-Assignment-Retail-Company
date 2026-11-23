"""Data backends. Both take BigQuery-dialect SQL and return a DataFrame.

LocalDB: synthetic copy of the 4 thelook tables in DuckDB, so everything runs without
GCP.
BigQueryDB: the real public dataset, with a hard cap on bytes billed.
"""

import random
from datetime import datetime, timedelta

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

from retail_agent.config import PII_COLUMNS, SQL, TABLES, Settings


def format_schema(tables: dict[str, list[tuple[str, str]]]) -> str:
    lines = []
    for table, columns in tables.items():
        lines.append(f"{table}:")
        for name, type_ in columns:
            lines.append(
                f"  {name} {type_}"
                + (" (PII, not queryable)" if name in PII_COLUMNS else "")
            )
    return "\n".join(lines)


class LocalDB:
    def __init__(self):
        self.con = duckdb.connect(":memory:")  # rebuilt each start (~3 s), never stale
        for name, df in _synthetic_thelook().items():
            self.con.from_df(df).create(name)

    def run(self, sql: str) -> pd.DataFrame:
        tree = sqlglot.parse_one(sql, read="bigquery")
        for table in tree.find_all(exp.Table):  # `project.dataset.table` -> table
            table.set("catalog", None)
            table.set("db", None)
        return self.con.execute(tree.sql(dialect="duckdb")).df()

    def schema(self) -> str:
        rows = self.con.execute(SQL["local_schema"]).fetchall()
        tables: dict[str, list[tuple[str, str]]] = {t: [] for t in TABLES}
        for table, column, type_ in rows:
            tables[table].append((column, type_))
        return format_schema(tables)


class BigQueryDB:
    def __init__(self, project: str | None, max_bytes_billed: int):
        from google.cloud import bigquery

        from retail_agent.bq_client import BigQueryRunner

        self.runner = BigQueryRunner(project_id=project)
        # BigQuery fails the job (free of charge) instead of scanning more than this.
        self.job_config = bigquery.QueryJobConfig(maximum_bytes_billed=max_bytes_billed)

    def run(self, sql: str) -> pd.DataFrame:
        job = self.runner.client.query(sql, job_config=self.job_config)
        try:
            df = job.result(timeout=60).to_dataframe()
        except TimeoutError:
            # otherwise the job keeps running (and using slots) after we gave up
            job.cancel()
            raise
        # for the trace
        df.attrs.update(job_id=job.job_id, bytes_processed=job.total_bytes_processed)
        return df

    def schema(self) -> str:
        return format_schema(
            {
                t: [(f["name"], f["type"]) for f in self.runner.get_table_schema(t)]
                for t in TABLES
            }
        )


def make_db(settings: Settings) -> LocalDB | BigQueryDB:
    if settings.data_backend == "bigquery":
        return BigQueryDB(settings.gcp_project, settings.max_bytes_billed)
    return LocalDB()


# ponytail: uniform random data, no seasonality; fine for testing SQL and flows. Load a
# sample of the real
# tables into DuckDB when local numbers must look real.
def _synthetic_thelook(
    n_users=2000, n_products=200, n_orders=6000, seed=42
) -> dict[str, pd.DataFrame]:
    rnd = random.Random(seed)
    start = datetime(2023, 1, 1)
    states = ["California", "Texas", "New York", "Florida", "Washington", "Illinois"]
    brands = [
        "Calvin Klein",
        "Levi's",
        "Columbia",
        "Carhartt",
        "Nike",
        "Allegra K",
        "Hanes",
        "Quiksilver",
    ]
    categories = [
        "Jeans",
        "Tops & Tees",
        "Outerwear & Coats",
        "Sweaters",
        "Active",
        "Accessories",
    ]

    def ts(days=1000):
        return start + timedelta(days=rnd.uniform(0, days))

    users = pd.DataFrame(
        {
            "id": i,
            "first_name": f"First{i}",
            "last_name": f"Last{i}",
            "email": f"user{i}@example.com",
            "age": rnd.randint(12, 70),
            "gender": rnd.choice("MF"),
            "state": rnd.choice(states),
            "street_address": f"{rnd.randint(1, 999)} Main St",
            "postal_code": f"{rnd.randint(10000, 99999)}",
            "city": "Springfield",
            "country": "United States",
            "latitude": rnd.uniform(25, 48),
            "longitude": rnd.uniform(-124, -70),
            "traffic_source": rnd.choice(
                ["Search", "Organic", "Facebook", "Email", "Display"]
            ),
            "created_at": ts(),
        }
        for i in range(1, n_users + 1)
    )
    products = pd.DataFrame(
        {
            "id": i,
            "cost": round((price := rnd.uniform(5, 150)) * rnd.uniform(0.3, 0.6), 2),
            "category": rnd.choice(categories),
            "name": f"Product {i}",
            "brand": rnd.choice(brands),
            "retail_price": round(price, 2),
            "department": rnd.choice(["Men", "Women"]),
        }
        for i in range(1, n_products + 1)
    )
    orders, items = [], []
    for order_id in range(1, n_orders + 1):
        user = users.iloc[rnd.randrange(n_users)]
        created = ts()
        status = rnd.choices(
            ["Complete", "Shipped", "Processing", "Cancelled", "Returned"],
            [6, 2, 1, 1, 1],
        )[0]
        n_items = rnd.randint(1, 4)
        orders.append(
            {
                "order_id": order_id,
                "user_id": int(user.id),
                "status": status,
                "gender": user.gender,
                "created_at": created,
                "returned_at": created + timedelta(days=10)
                if status == "Returned"
                else None,
                "shipped_at": created + timedelta(days=1)
                if status != "Processing"
                else None,
                "delivered_at": created + timedelta(days=4)
                if status in ("Complete", "Returned")
                else None,
                "num_of_item": n_items,
            }
        )
        for _ in range(n_items):
            product = products.iloc[rnd.randrange(n_products)]
            items.append(
                {
                    "id": len(items) + 1,
                    "order_id": order_id,
                    "user_id": int(user.id),
                    "product_id": int(product.id),
                    "status": status,
                    "created_at": created,
                    "sale_price": product.retail_price,
                }
            )
    return {
        "users": users,
        "products": products,
        "orders": pd.DataFrame(orders),
        "order_items": pd.DataFrame(items),
    }
