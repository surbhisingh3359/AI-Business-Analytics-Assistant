import pandas as pd
import pytest

from src.data_cleaner import clean_data, fill_missing_values


def test_clean_data():
    df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1002, None],
            "product": ["Laptop", "Mouse", "Mouse", None],
            "quantity": [2, 5, 5, None],
            "empty_column": [None, None, None, None],
        }
    )

    cleaned_df = clean_data(df)

    assert len(cleaned_df) == 2
    assert "empty_column" not in cleaned_df.columns
    assert cleaned_df["order_id"].tolist() == [1001, 1002]


def test_clean_data_does_not_modify_original():
    df = pd.DataFrame(
        {
            "product": ["Laptop", "Mouse", "Mouse"],
        }
    )

    original_df = df.copy()

    clean_data(df)

    pd.testing.assert_frame_equal(df, original_df)


def test_clean_data_invalid_input():
    with pytest.raises(TypeError):
        clean_data("not a dataframe")


def test_fill_missing_numeric_values():
    df = pd.DataFrame(
        {
            "age": [20, 30, None, 40],
        }
    )

    cleaned_df = fill_missing_values(df)

    assert cleaned_df["age"].isnull().sum() == 0
    assert cleaned_df["age"].iloc[2] == 30


def test_fill_missing_categorical_values():
    df = pd.DataFrame(
        {
            "city": ["Delhi", None, "Mumbai"],
        }
    )

    cleaned_df = fill_missing_values(df)

    assert cleaned_df["city"].isnull().sum() == 0
    assert cleaned_df["city"].iloc[1] == "Unknown"


def test_fill_missing_invalid_input():
    with pytest.raises(TypeError):
        fill_missing_values("not a dataframe")