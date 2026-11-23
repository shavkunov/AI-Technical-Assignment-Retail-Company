import pytest


@pytest.fixture
def db(local_db):
    return local_db


def test_local_tables_exist(db):
    for table in ("orders", "order_items", "products", "users"):
        assert (
            len(
                db.run(
                    "SELECT * FROM "
                    f"`bigquery-public-data.thelook_ecommerce.{table}` "
                    "LIMIT 5"
                )
            )
            == 5
        )


def test_local_runs_bigquery_dialect(db):
    df = db.run(
        "SELECT FORMAT_DATE('%Y-%m', "
        "DATE_TRUNC(DATE(created_at), MONTH)) AS m, "
        "SUM(sale_price) AS rev "
        "FROM "
        "`bigquery-public-data.thelook_ecommerce.order_items` "
        "GROUP BY m ORDER BY m"
    )
    assert len(df) > 12 and df["m"].iloc[0][4] == "-"


def test_local_items_reference_real_products_and_users(db):
    orphans = db.run(
        "SELECT COUNT(*) AS n FROM order_items oi "
        "LEFT JOIN products p ON p.id = oi.product_id LEFT "
        "JOIN users u ON u.id = oi.user_id "
        "WHERE p.id IS NULL OR u.id IS NULL"
    )
    assert orphans["n"].iloc[0] == 0


def test_schema_marks_pii(db):
    schema = db.schema()
    assert "users" in schema and "email" in schema
    assert next(line for line in schema.splitlines() if "email" in line).endswith(
        "(PII, not queryable)"
    )


def test_bigquery_job_is_cancelled_on_timeout():
    from types import SimpleNamespace

    from retail_agent.db import BigQueryDB

    cancelled = []

    class Job:
        def result(self, timeout):
            raise TimeoutError("slow")

        def cancel(self):
            cancelled.append(True)

    db = BigQueryDB.__new__(BigQueryDB)
    db.runner = SimpleNamespace(
        client=SimpleNamespace(query=lambda sql, job_config: Job())
    )
    db.job_config = None
    with pytest.raises(TimeoutError):
        db.run("SELECT 1")
    assert cancelled == [True]


def test_bigquery_run_reports_job_id_and_bytes():
    from types import SimpleNamespace

    import pandas as pd

    from retail_agent.db import BigQueryDB

    job = SimpleNamespace(
        job_id="job-9",
        total_bytes_processed=42,
        result=lambda timeout: SimpleNamespace(
            to_dataframe=lambda: pd.DataFrame({"n": [1]})
        ),
    )
    db = BigQueryDB.__new__(BigQueryDB)
    db.runner = SimpleNamespace(
        client=SimpleNamespace(query=lambda sql, job_config: job)
    )
    db.job_config = None
    assert db.run("SELECT 1").attrs == {"job_id": "job-9", "bytes_processed": 42}
