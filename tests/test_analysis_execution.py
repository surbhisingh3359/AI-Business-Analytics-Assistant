import pytest

from src.analysis_execution import (
    build_execution_plan,
    execute_analysis,
)


def test_build_execution_plan_for_group_comparison():

    decision = {
        "decision": (
            "Investigate the difference between the observed groups."
        ),
        "recommended_analysis": "group_comparison",
        "evidence": [
            "Electronics has higher average revenue than Furniture."
        ],
        "metrics": [
            "revenue"
        ],
        "type": "group_comparison",
    }

    result = build_execution_plan(decision)

    assert result["analysis_type"] == "group_comparison"
    assert result["metric"] == "revenue"
    assert result["status"] == "planned"


def test_build_execution_plan_for_correlation():

    decision = {
        "decision": (
            "Investigate the relationship between the observed variables."
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

    result = build_execution_plan(decision)

    assert result["analysis_type"] == "correlation_analysis"
    assert result["status"] == "planned"


def test_build_execution_plan_for_trend():

    decision = {
        "decision": (
            "Investigate the observed trend over time."
        ),
        "recommended_analysis": "trend_analysis",
        "evidence": [
            "Revenue increased over the observed period."
        ],
        "metrics": [
            "revenue"
        ],
        "type": "trend",
    }

    result = build_execution_plan(decision)

    assert result["analysis_type"] == "trend_analysis"
    assert result["status"] == "planned"


def test_build_execution_plan_for_risk():

    decision = {
        "decision": (
            "Investigate the observed risk."
        ),
        "recommended_analysis": "risk_analysis",
        "evidence": [
            "Revenue concentration is high."
        ],
        "metrics": [
            "revenue"
        ],
        "type": "risk",
    }

    result = build_execution_plan(decision)

    assert result["analysis_type"] == "risk_analysis"
    assert result["status"] == "planned"


def test_build_execution_plan_rejects_invalid_input():

    with pytest.raises(TypeError):

        build_execution_plan("invalid")


def test_build_execution_plan_requires_analysis_type():

    decision = {
        "decision": "Investigate this finding.",
        "evidence": ["Some evidence."],
    }

    with pytest.raises(ValueError):

        build_execution_plan(decision)


def test_execute_analysis_requires_dataframe():

    plan = {
        "analysis_type": "group_comparison",
        "metric": "revenue",
        "status": "planned",
    }

    with pytest.raises(TypeError):

        execute_analysis(
            None,
            plan,
        )