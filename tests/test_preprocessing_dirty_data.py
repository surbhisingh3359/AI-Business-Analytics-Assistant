import pandas as pd

from src.preprocessing import preprocess_data


def test_preprocess_dirty_data(tmp_path):
    dirty_df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1002, None],
            "product": ["Laptop", "Mouse", "Mouse", None],
            "quantity": [2, 5, 5, None],
            "empty_column": [None, None, None, None],
        }
    )

    file_path = tmp_path / "dirty_data.csv"
    dirty_df.to_csv(file_path, index=False)

    result = preprocess_data(str(file_path))

    cleaned_df = result["data"]
    quality_report = result["quality_report"]

    assert "empty_column" not in cleaned_df.columns
    assert len(cleaned_df) == 2

    assert quality_report["before"]["rows"] == 4
    assert quality_report["after"]["rows"] == 2

    assert quality_report["before"]["duplicates"] == 1
    assert quality_report["after"]["duplicates"] == 0

    assert quality_report["before"]["missing_values"] == 7
    assert quality_report["after"]["missing_values"] == 0