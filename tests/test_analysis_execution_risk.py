import pandas as pd
import pytest

from src.analysis_execution import execute_analysis


def test_execute_risk_analysis():

    dataframe = pd.DataFrame(
        {
            "product": [
                "A",
                "B",
                "C",
                "D",
            ],
            "revenue": [
                7000,
                1000,
                1000,
                1000,
            ],
        }
    )

    plan = {
        "analysis_type": "risk_analysis",
        "metric": "revenue",
        "status": "planned",
    }

    result = execute_analysis(
        dataframe,
        plan,
    )

    assert result["analysis_type"] == "risk_analysis"
    assert result["status"] == "completed"
    assert result["metric"] == "revenue"

    assert "total" in result
    assert result["total"] == 10000

    assert "highest_value" in result
    assert result["highest_value"] == 7000

    assert "highest_share" in result
    assert result["highest_share"] == pytest.approx(0.70)

    assert "risk_level" in result
    
def test_risk_requires_metric():

    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    plan = {
        "analysis_type": "risk_analysis",
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )


def test_risk_requires_existing_metric():

    dataframe = pd.DataFrame(
        {
            "revenue": [100, 200, 300],
        }
    )

    plan = {
        "analysis_type": "risk_analysis",
        "metric": "profit",
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )


def test_risk_requires_numeric_metric():

    dataframe = pd.DataFrame(
        {
            "revenue": [
                "high",
                "medium",
                "low",
            ]
        }
    )

    plan = {
        "analysis_type": "risk_analysis",
        "metric": "revenue",
        "status": "planned",
    }

    with pytest.raises(ValueError):

        execute_analysis(
            dataframe,
            plan,
        )