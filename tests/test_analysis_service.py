from unittest.mock import patch

import pandas as pd
import pytest

from src.analysis_service import run_analysis


def test_run_analysis_returns_complete_result():

    fake_response = (
        "The business generated 136,000 revenue "
        "from 5 orders. "
        "The company should diversify its revenue sources."
    )

    with patch(
        "src.analysis_service.LLMClient.generate",
        return_value=fake_response,
    ):

        result = run_analysis(
            "data/raw/sample_sales.csv"
        )

    assert isinstance(result, dict)

    assert "data" in result
    assert "quality_report" in result
    assert "context" in result
    assert "prompt" in result
    assert "analysis" in result
    assert "validation" in result

    assert isinstance(
        result["data"],
        pd.DataFrame,
    )

    assert isinstance(
        result["prompt"],
        str,
    )

    assert result["analysis"] == fake_response

    assert isinstance(
        result["validation"],
        dict,
    )


def test_run_analysis_uses_custom_model():

    fake_response = (
        "The business generated 136,000 revenue "
        "from 5 orders. "
        "The company should diversify its revenue sources."
    )

    with patch(
        "src.analysis_service.LLMClient.generate",
        return_value=fake_response,
    ) as mock_generate:

        result = run_analysis(
            "data/raw/sample_sales.csv",
            model="custom-model",
        )

    assert result["analysis"] == fake_response

    mock_generate.assert_called_once()


def test_run_analysis_rejects_non_string_file_path():

    with pytest.raises(TypeError):
        run_analysis(None)


def test_run_analysis_rejects_empty_file_path():

    with pytest.raises(ValueError):
        run_analysis("")


def test_run_analysis_rejects_non_string_model():

    with pytest.raises(TypeError):
        run_analysis(
            "data/raw/sample_sales.csv",
            model=None,
        )


def test_run_analysis_rejects_empty_model():

    with pytest.raises(ValueError):
        run_analysis(
            "data/raw/sample_sales.csv",
            model="",
        )