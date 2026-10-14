import pytest

from src.decision_engine import (
    build_decision,
    build_decision_layer,
)


def test_build_group_comparison_decision():

    insight = {
        "insight": (
            "Average revenue differs across countries."
        ),
        "type": "group_comparison",
        "metric": "revenue",
        "category": "country",
        "confidence": "high",
    }

    result = build_decision(
        insight
    )

    assert (
        "differences in revenue"
        in result["decision"]
    )

    assert result["recommended_analysis"] == (
        "group_comparison"
    )

    assert result["metrics"] == [
        "revenue"
    ]

    assert result["dimensions"] == [
        "country"
    ]

    assert result["confidence"] == "high"


def test_build_correlation_decision():

    insight = {
        "insight": (
            "Quantity and revenue are strongly correlated."
        ),
        "type": "correlation",
        "confidence": "medium",
    }

    result = build_decision(
        insight
    )

    assert result["recommended_analysis"] == (
        "correlation_analysis"
    )

    assert (
        "causation"
        in result["limitations"][0]
    )


def test_build_trend_decision():

    insight = {
        "insight": (
            "Revenue increased over time."
        ),
        "type": "trend",
    }

    result = build_decision(
        insight
    )

    assert result["recommended_analysis"] == (
        "time_analysis"
    )

    assert (
        "continue"
        in result["limitations"][0]
    )


def test_build_risk_decision():

    insight = {
        "insight": (
            "Revenue concentration is high."
        ),
        "type": "risk",
        "confidence": "high",
    }

    result = build_decision(
        insight
    )

    assert result["recommended_analysis"] == (
        "risk_analysis"
    )

    assert result["confidence"] == "high"


def test_build_decision_rejects_invalid_input():

    with pytest.raises(TypeError):
        build_decision("invalid")


def test_build_decision_requires_text():

    with pytest.raises(ValueError):
        build_decision(
            {
                "type": "risk"
            }
        )


def test_build_decision_layer():

    insights = [
        {
            "insight": "Revenue differs by country.",
            "type": "group_comparison",
            "metric": "revenue",
            "category": "country",
        },
        {
            "insight": "Revenue and quantity are correlated.",
            "type": "correlation",
        },
    ]

    result = build_decision_layer(
        insights
    )

    assert len(result) == 2

    assert result[0]["source_type"] == (
        "group_comparison"
    )

    assert result[1]["source_type"] == (
        "correlation"
    )


def test_decision_layer_removes_duplicates():

    insights = [
        {
            "insight": "Revenue differs by country.",
            "type": "group_comparison",
        },
        {
            "insight": "Revenue differs by country.",
            "type": "group_comparison",
        },
    ]

    result = build_decision_layer(
        insights
    )

    assert len(result) == 1


def test_decision_layer_limits_results():

    insights = [
        {
            "insight": f"Insight {i}",
            "type": "risk",
        }
        for i in range(10)
    ]

    result = build_decision_layer(
        insights,
        max_decisions=3,
    )

    assert len(result) == 3


def test_invalid_max_decisions():

    with pytest.raises(ValueError):
        build_decision_layer(
            [],
            max_decisions=0,
        )