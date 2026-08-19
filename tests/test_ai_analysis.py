import pandas as pd
import pytest
from unittest.mock import patch

from src.ai_analysis import analyze_business_data


def test_analyze_business_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        analyze_business_data("does_not_exist.csv")


def test_analyze_business_data_includes_validation():
    fake_response = (
        "The business generated 136000 revenue from 5 orders. "
        "The company should diversify its revenue sources."
    )

    with patch(
        "src.ai_analysis.LLMClient.generate",
        return_value=fake_response,
    ):
        result = analyze_business_data(
            "data/raw/sample_sales.csv"
        )

    assert "validation" in result
    assert result["validation"]["is_valid"] is True
    assert result["validation"]["has_recommendations"] is True


def test_analyze_business_data_validation_detects_missing_metrics():
    fake_response = (
        "The company should diversify its revenue sources."
    )

    with patch(
        "src.ai_analysis.LLMClient.generate",
        return_value=fake_response,
    ):
        result = analyze_business_data(
            "data/raw/sample_sales.csv"
        )

    assert result["validation"]["is_valid"] is False
    assert len(result["validation"]["warnings"]) > 0