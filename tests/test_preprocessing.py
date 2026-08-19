from unittest import result

import pandas as pd

from src.preprocessing import preprocess_data


def test_preprocess_data():
    result = preprocess_data("data/raw/sample_sales.csv")

    assert "data" in result
    assert "profile" in result
    assert "schema" in result

    assert isinstance(result["data"], pd.DataFrame)

    assert result["profile"]["rows"] == 5
    assert result["profile"]["columns"] == 5

    assert result["schema"]["order_id"]["role"] == "identifier"
    assert result["schema"]["quantity"]["role"] == "numeric"
    assert "quality_report" in result

    assert result["quality_report"]["before"]["rows"] == 5
    assert result["quality_report"]["after"]["rows"] == 5