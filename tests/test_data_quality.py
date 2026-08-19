import pandas as pd
import pytest

from src.data_quality import generate_quality_report


def test_generate_quality_report():
    before_df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1002, None],
            "product": ["Laptop", "Mouse", "Mouse", None],
            "quantity": [2, 5, 5, None],
        }
    )

    after_df = pd.DataFrame(
        {
            "order_id": [1001, 1002],
            "product": ["Laptop", "Mouse"],
            "quantity": [2, 5],
        }
    )

    report = generate_quality_report(before_df, after_df)

    assert report["before"]["rows"] == 4
    assert report["after"]["rows"] == 2

    assert report["before"]["missing_values"] == 3
    assert report["after"]["missing_values"] == 0

    assert report["changes"]["rows_removed"] == 2
    assert report["changes"]["missing_values_resolved"] == 3


def test_quality_report_invalid_before_input():
    after_df = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(TypeError):
        generate_quality_report("not a dataframe", after_df)


def test_quality_report_invalid_after_input():
    before_df = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(TypeError):
        generate_quality_report(before_df, "not a dataframe")