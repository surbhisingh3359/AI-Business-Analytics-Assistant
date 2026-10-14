import pandas as pd
import pytest

from src.analysis_execution import execute_analysis


def test_execute_correlation_analysis():

    dataframe = pd.DataFrame(
        {
            "revenue": [
                100,
                200,
                300,
                400,
                500,
            ],
            "quantity": [
                10,
                20,
                30,
                40,
                50,
            ],
        }
    )

    plan = {
        "analysis_type": "correlation_analysis",
        "metrics": [
            "revenue",
            "quantity",
        ],
        "status": "planned",
    }

    result = execute_analysis(
        dataframe,
        plan,
    )

    assert result["analysis_type"] == "correlation_analysis"
    assert result["status"] == "completed"

    assert result["metrics"] == [
        "revenue",
        "quantity",
    ]

    assert "correlation" in result
    assert result["correlation"] == pytest.approx(1.0)


def test_correlation_requires_two_metrics():

    dataframe = pd.DataFrame(
        {
            "revenue": [
                100,
                200,
                300,
            ],
        }
    )

    plan = {
        "analysis_type": "correlation_analysis",
        "metrics": [
            "revenue",
        ],
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )


def test_correlation_requires_existing_metrics():

    dataframe = pd.DataFrame(
        {
            "revenue": [
                100,
                200,
                300,
            ],
            "quantity": [
                10,
                20,
                30,
            ],
        }
    )

    plan = {
        "analysis_type": "correlation_analysis",
        "metrics": [
            "revenue",
            "profit",
        ],
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )


def test_correlation_requires_numeric_metrics():

    dataframe = pd.DataFrame(
        {
            "revenue": [
                "high",
                "medium",
                "low",
            ],
            "quantity": [
                10,
                20,
                30,
            ],
        }
    )

    plan = {
        "analysis_type": "correlation_analysis",
        "metrics": [
            "revenue",
            "quantity",
        ],
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )