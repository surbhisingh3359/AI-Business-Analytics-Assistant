import pytest

from src.insight_ranker import rank_insights


def test_ranks_high_priority_insights_first():

    insights = [
        {
            "insight": "Sales increased by 5%.",
            "impact": 1,
        },
        {
            "insight": "Revenue concentration is high.",
            "impact": 10,
        },
        {
            "insight": "Average order value increased.",
            "impact": 5,
        },
    ]

    result = rank_insights(insights)

    assert result[0]["insight"] == (
        "Revenue concentration is high."
    )


def test_removes_duplicate_insights():

    insights = [
        {
            "insight": "IT has the highest average salary.",
            "impact": 5,
        },
        {
            "insight": "IT has the highest average salary.",
            "impact": 3,
        },
    ]

    result = rank_insights(insights)

    assert len(result) == 1


def test_limits_number_of_insights():

    insights = [
        {
            "insight": f"Insight number {i}",
            "impact": i,
        }
        for i in range(10)
    ]

    result = rank_insights(
        insights,
        max_insights=3,
    )

    assert len(result) == 3


def test_invalid_input():

    with pytest.raises(TypeError):
        rank_insights("invalid")


def test_invalid_max_insights():

    with pytest.raises(ValueError):
        rank_insights([], max_insights=0)