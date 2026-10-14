import pandas as pd
import pytest

from src.analysis_execution import (
    build_execution_plan,
    execute_analysis,
)


def test_correlation_pipeline():

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

    decision = {
        "decision": (
            "Investigate the relationship between revenue and quantity."
        ),
        "recommended_analysis": "correlation_analysis",
        "evidence": [
            "Revenue and quantity show a strong relationship."
        ],
        "metrics": [
            "revenue",
            "quantity",
        ],
        "type": "correlation",
    }

    plan = build_execution_plan(decision)

    assert plan["analysis_type"] == "correlation_analysis"
    assert plan["status"] == "planned"

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

    assert result["correlation"] == pytest.approx(1.0)


def test_trend_pipeline():

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

    decision = {
        "decision": (
            "Investigate the observed revenue trend over time."
        ),
        "recommended_analysis": "time_analysis",
        "evidence": [
            "Revenue increased over time."
        ],
        "metrics": [
            "revenue",
        ],
        "type": "trend",
    }

    plan = build_execution_plan(decision)

    assert plan["analysis_type"] == "time_analysis"
    assert plan["metric"] == "revenue"
    assert plan["status"] == "planned"

    result = execute_analysis(
        dataframe,
        plan,
    )

    assert result["analysis_type"] == "time_analysis"
    assert result["status"] == "completed"
    assert result["metric"] == "revenue"
    assert result["direction"] == "increasing"


def test_risk_pipeline():

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

    decision = {
        "decision": (
            "Investigate the concentration risk in revenue."
        ),
        "recommended_analysis": "risk_analysis",
        "evidence": [
            "Revenue is highly concentrated in one product."
        ],
        "metrics": [
            "revenue",
        ],
        "type": "risk",
    }

    plan = build_execution_plan(decision)

    assert plan["analysis_type"] == "risk_analysis"
    assert plan["metric"] == "revenue"
    assert plan["status"] == "planned"

    result = execute_analysis(
        dataframe,
        plan,
    )

    assert result["analysis_type"] == "risk_analysis"
    assert result["status"] == "completed"
    assert result["metric"] == "revenue"

    assert result["total"] == 10000
    assert result["highest_value"] == 7000
    assert result["highest_share"] == pytest.approx(0.70)
    assert "risk_level" in result