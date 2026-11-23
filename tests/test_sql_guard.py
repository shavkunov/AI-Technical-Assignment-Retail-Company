import pytest

from retail_agent.sql_guard import SqlRejected, guard

T = "`bigquery-public-data.thelook_ecommerce`"


@pytest.mark.parametrize(
    "sql",
    [
        f"DELETE FROM {T}.orders WHERE true",
        f"DROP TABLE {T}.orders",
        f"INSERT INTO {T}.orders (order_id) VALUES (1)",
        f"SELECT 1 FROM {T}.orders; DROP TABLE {T}.orders",
        "UPDATE orders SET status = 'x' WHERE true",
    ],
)
def test_rejects_non_select(sql):
    with pytest.raises(SqlRejected):
        guard(sql, brands=None)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM `bigquery-public-data.other_dataset.secrets`",
        "SELECT * FROM `my-project.thelook_ecommerce.orders`",
        "SELECT * FROM INFORMATION_SCHEMA.TABLES",
        "SELECT * FROM events",
    ],
)
def test_rejects_unknown_table(sql):
    with pytest.raises(SqlRejected, match="table"):
        guard(sql, brands=None)


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT email FROM {T}.users",
        f"SELECT u.email AS contact FROM {T}.users u",
        f"WITH x AS (SELECT first_name FROM {T}.users) SELECT COUNT(*) FROM x",
        f"SELECT UPPER(last_name) FROM {T}.users",
        f"SELECT id FROM {T}.users WHERE street_address LIKE '%Main%'",
    ],
)
def test_rejects_pii_column(sql):
    with pytest.raises(SqlRejected, match="PII"):
        guard(sql, brands=None)


@pytest.mark.parametrize(
    "sql", [f"SELECT * FROM {T}.users", f"SELECT u.* FROM {T}.users u"]
)
def test_rejects_star_on_users(sql):
    with pytest.raises(SqlRejected, match="PII"):
        guard(sql, brands=None)


def test_star_allowed_on_non_pii_tables():
    guard(f"SELECT * FROM {T}.products", brands=None)


def test_adds_limit():
    assert "LIMIT 200" in guard(f"SELECT state FROM {T}.users", brands=None)


def test_keeps_smaller_existing_limit():
    assert "LIMIT 5" in guard(f"SELECT state FROM {T}.users LIMIT 5", brands=None)


def test_admin_not_filtered():
    assert "brand IN" not in guard(f"SELECT id FROM {T}.products", brands=None)


@pytest.fixture
def db(local_db):
    return local_db


ALLOWED = ["Calvin Klein", "Levi's"]


def test_brand_filter_on_products(db):
    df = db.run(guard(f"SELECT DISTINCT brand FROM {T}.products", brands=ALLOWED))
    assert set(df["brand"]) == set(ALLOWED)


def test_brand_filter_on_items_without_join(db):
    sql = guard(f"SELECT DISTINCT product_id FROM {T}.order_items", brands=ALLOWED)
    ids = set(db.run(sql)["product_id"])
    allowed_ids = set(
        db.run("SELECT id FROM products WHERE brand IN ('Calvin Klein', \"Levi's\")")[
            "id"
        ]
    )
    assert ids and ids <= allowed_ids


def test_brand_filter_on_orders_and_users(db):
    total_orders = db.run(f"SELECT COUNT(*) AS n FROM {T}.orders")["n"][0]
    total_users = db.run(f"SELECT COUNT(*) AS n FROM {T}.users")["n"][0]
    sql = guard(
        "SELECT COUNT(DISTINCT o.order_id) AS n_orders, "
        "COUNT(DISTINCT u.id) AS n_users "
        f"FROM {T}.orders o JOIN {T}.users u ON u.id = o.user_id",
        brands=ALLOWED,
    )
    row = db.run(sql).iloc[0]
    assert 0 < row.n_orders < total_orders and 0 < row.n_users < total_users


def test_brand_filter_inside_cte(db):
    sql = guard(
        f"WITH p AS (SELECT brand FROM {T}.products) SELECT DISTINCT brand FROM p",
        brands=ALLOWED,
    )
    assert set(db.run(sql)["brand"]) == set(ALLOWED)


@pytest.mark.parametrize(
    "sql",
    [f"SELECT u FROM {T}.users u", f"SELECT TO_JSON_STRING(u) FROM {T}.users AS u"],
)
def test_rejects_whole_users_row_as_struct(sql):
    with pytest.raises(SqlRejected, match="PII"):
        guard(sql, brands=None)


def test_brand_names_are_escaped(db):
    sql = guard(f"SELECT COUNT(*) AS n FROM {T}.products", brands=["x') OR TRUE OR ('"])
    assert db.run(sql)["n"][0] == 0


def test_syntax_error_is_rejected_with_message():
    with pytest.raises(SqlRejected, match="syntax"):
        guard("SELEC brand FROM products", brands=None)


@pytest.mark.parametrize(
    "sql",
    [
        "WITH products AS (SELECT * FROM products) SELECT DISTINCT brand FROM products",
        f"WITH users AS (SELECT id FROM {T}.users) SELECT COUNT(*) FROM users",
    ],
)
def test_rejects_cte_shadowing_a_table(sql):
    with pytest.raises(SqlRejected, match="shadow"):
        guard(sql, brands=ALLOWED)


@pytest.mark.parametrize(
    "sql",
    [
        f"SELECT TO_JSON_STRING(U) FROM {T}.users u",
        f"SELECT TO_JSON_STRING(u) FROM {T}.users AS U",
        f"SELECT TO_JSON_STRING(USERS) FROM {T}.users",
        f"SELECT ARRAY_AGG(U LIMIT 3) FROM {T}.users u",
    ],
)
def test_rejects_whole_users_row_any_alias_case(sql):
    with pytest.raises(SqlRejected, match="PII"):
        guard(sql, brands=ALLOWED)


def test_syntax_error_message_is_plain_text():
    with pytest.raises(SqlRejected) as e:
        guard(f"SELECT (1 FROM {T}.products", brands=None)
    assert "\x1b" not in str(e.value) and "line 1" in str(e.value)


@pytest.mark.parametrize(
    "sql",
    [
        # any users.* is a whole row
        f"SELECT TO_JSON_STRING(STRUCT(u.*)) FROM {T}.users u",
        # lost by the brand rewrite
        f"SELECT id FROM {T}.products FOR SYSTEM_TIME AS OF TIMESTAMP '2020-01-01'",
        f"SELECT id FROM {T}.products TABLESAMPLE SYSTEM (10 PERCENT)",
    ],
)
def test_rejects_row_star_time_travel_and_sampling(sql):
    with pytest.raises(SqlRejected):
        guard(sql, brands=["Calvin Klein"])


def test_rejects_cte_shadowing_a_table_in_any_case():
    # DuckDB/BigQuery resolve names case-insensitively: `Products` would be the
    # unfiltered table.
    with pytest.raises(SqlRejected):
        guard(
            "WITH Products AS (SELECT * FROM Products) SELECT brand FROM Products",
            brands=["Calvin Klein"],
        )


def test_accepts_ansi_doubled_apostrophe(local_db):
    # Gemini often writes 'Levi''s' (ANSI); BigQuery needs 'Levi\'s'. Fixing it here
    # saves a failed query + model call.
    sql = guard(
        f"SELECT brand FROM {T}.products WHERE brand IN ('Levi''s', '') LIMIT 5",
        brands=None,
    )
    assert list(local_db.run(sql)["brand"].unique()) == ["Levi's"]


def test_unclosed_string_is_a_syntax_error_for_the_model():
    # sqlglot raises TokenError here, not ParseError
    with pytest.raises(SqlRejected, match="syntax"):
        guard(f"SELECT 1 FROM {T}.products WHERE brand = 'Levi's'", brands=None)
