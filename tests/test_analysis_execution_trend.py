import pandas as pd
import pytest

from src.analysis_execution import execute_analysis


def test_execute_trend_analysis():

    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                    "2026-03-01",
                    "2026-04-01",
                ]
            ),
            "revenue": [
                1000,
                1200,
                1500,
                1800,
            ],
        }
    )

    plan = {
        "analysis_type": "trend_analysis",
        "metric": "revenue",
        "status": "planned",
    }

    result = execute_analysis(
        dataframe,
        plan,
    )

    assert result["analysis_type"] == "trend_analysis"
    assert result["status"] == "completed"
    assert result["metric"] == "revenue"

    assert result["direction"] == "increasing"
    assert result["first_value"] == 1000
    assert result["last_value"] == 1800
    assert result["change"] == 800


def test_trend_requires_metric():

    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                ]
            ),
            "revenue": [
                100,
                200,
            ],
        }
    )

    plan = {
        "analysis_type": "trend_analysis",
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )


def test_trend_requires_time_column():

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
        "analysis_type": "trend_analysis",
        "metric": "revenue",
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )


def test_trend_requires_numeric_metric():

    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                    "2026-03-01",
                ]
            ),
            "revenue": [
                "high",
                "medium",
                "low",
            ],
        }
    )

    plan = {
        "analysis_type": "trend_analysis",
        "metric": "revenue",
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )