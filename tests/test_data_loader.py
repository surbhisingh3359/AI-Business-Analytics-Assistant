from pathlib import Path

import pandas as pd

from src.data_loader import load_data
import pytest

SAMPLE_FILE = Path("data/raw/sample_sales.csv")

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("data/raw/does_not_exist.csv")


def test_load_csv():
    df = load_data(SAMPLE_FILE)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == [
        "order_id",
        "product",
        "category",
        "quantity",
        "price",
    ]

def test_unsupported_file_type(tmp_path):
    unsupported_file = tmp_path / "sales.json"
    unsupported_file.write_text("{}")

    with pytest.raises(ValueError):
        load_data(unsupported_file)