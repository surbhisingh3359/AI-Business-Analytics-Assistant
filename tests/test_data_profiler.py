import pandas as pd
import pytest

from src.data_profiler import profile_data


def test_profile_data():
    df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1003],
            "product": ["Laptop", "Mouse", "Keyboard"],
            "quantity": [2, 5, 3],
            "price": [55000, 800, 1500],
        }
    )

    profile = profile_data(df)

    assert profile["rows"] == 3
    assert profile["columns"] == 4
    assert profile["column_names"] == [
        "order_id",
        "product",
        "quantity",
        "price",
    ]
    assert profile["missing_values"]["quantity"] == 0
    assert profile["duplicate_rows"] == 0

def test_profile_data_invalid_input():
    with pytest.raises(TypeError):
        profile_data("not a dataframe")