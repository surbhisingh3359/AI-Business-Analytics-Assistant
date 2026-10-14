import pandas as pd
import pytest

from src.analysis_execution import execute_analysis


def test_execute_group_comparison():

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

    plan = {
        "analysis_type": "group_comparison",
        "metric": "revenue",
        "status": "planned",
    }

    result = execute_analysis(
        dataframe,
        plan,
    )

    assert result["analysis_type"] == "group_comparison"
    assert result["status"] == "completed"
    assert result["metric"] == "revenue"

    assert result["highest_group"] == "Electronics"
    assert result["lowest_group"] == "Furniture"

    assert result["group_values"]["Electronics"] == 2500
    assert result["group_values"]["Furniture"] == 1200


def test_group_comparison_requires_metric():

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

    plan = {
        "analysis_type": "group_comparison",
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )


def test_group_comparison_requires_group_column():

    dataframe = pd.DataFrame(
        {
            "revenue": [
                100,
                200,
            ],
        }
    )

    plan = {
        "analysis_type": "group_comparison",
        "metric": "revenue",
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )


def test_group_comparison_requires_numeric_metric():

    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
            ],
            "revenue": [
                "high",
                "low",
            ],
        }
    )

    plan = {
        "analysis_type": "group_comparison",
        "metric": "revenue",
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )