import pandas as pd
import pytest

from src.business_analysis_pipeline import (
    run_business_analysis,
    run_business_prediction,
)


def test_run_business_analysis():

    dataframe = pd.DataFrame(
        {
            "category": [
                "Electronics",
                "Electronics",
                "Furniture",
                "Furniture",
            ],
            "revenue": [
                1000,
                1500,
                500,
                700,
            ],
        }
    )

    result = run_business_analysis(
        dataframe
    )

    assert isinstance(
        result,
        dict,
    )

    assert result["status"] in {
        "completed",
        "completed_with_errors",
    }

    assert result["dataset"]["rows"] == 4

    assert "semantic_roles" in result
    assert "analysis_plan" in result
    assert "analytics" in result
    assert "insights" in result
    assert "decisions" in result
    assert "execution_results" in result

    assert result["insight_count"] >= 1

    assert result["decision_count"] >= 1


def test_run_business_analysis_rejects_invalid_dataframe():

    with pytest.raises(TypeError):

        run_business_analysis(
            "not a dataframe"
        )


def test_run_business_analysis_rejects_empty_dataframe():

    dataframe = pd.DataFrame()

    with pytest.raises(ValueError):

        run_business_analysis(
            dataframe
        )


def test_run_business_analysis_validates_limits():

    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
            ],
            "revenue": [
                100,
                200,
            ],
        }
    )

    with pytest.raises(ValueError):

        run_business_analysis(
            dataframe,
            max_insights=0,
        )

    with pytest.raises(ValueError):

        run_business_analysis(
            dataframe,
            max_decisions=0,
        )

def test_run_business_analysis_exposes_predictive_options():

    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
            ],
            "quantity": [
                1,
                2,
                3,
                4,
                5,
            ],
            "price": [
                100,
                200,
                300,
                400,
                500,
            ],
        }
    )

    result = run_business_analysis(
        dataframe
    )

    options = result[
        "predictive_options"
    ]

    assert options["available"] is True

    assert "price" in (
        options["regression_candidates"]
    )

    assert "quantity" in (
        options["regression_candidates"]
    )

    assert "category" in (
        options["classification_candidates"]
    )


def test_run_business_prediction():

    dataframe = pd.DataFrame(
        {
            "quantity": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "price": [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
            ],
        }
    )

    result = run_business_prediction(
        dataframe,
        "price",
    )

    assert result["status"] == "completed"

    assert result["task"] == "regression"

    assert result["target"] == "price"

    assert "metrics" in result

    assert "predictions" in result

    assert "actual_values" in result


def test_run_business_prediction_requires_target():

    dataframe = pd.DataFrame(
        {
            "price": [
                100,
                200,
                300,
                400,
                500,
            ]
        }
    )

    with pytest.raises(
        ValueError
    ):

        run_business_prediction(
            dataframe,
            "missing",
        )