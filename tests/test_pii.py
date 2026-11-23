import pandas as pd

from retail_agent.pii import mask_frame


def test_drops_pii_columns():
    df = pd.DataFrame(
        {
            "email": ["a@b.com"],
            "first_name": ["Ann"],
            "state": ["Texas"],
            "spend": [10.0],
        }
    )
    assert list(mask_frame(df).columns) == ["state", "spend"]


def test_pseudonym_stable_and_not_raw():
    df = pd.DataFrame({"user_id": [17, 17, 42], "spend": [1, 2, 3]})
    out = mask_frame(df)["user_id"].tolist()
    assert out[0] == out[1] != out[2]
    assert all(v.startswith("C-") and "17" not in v and "42" not in v for v in out)
