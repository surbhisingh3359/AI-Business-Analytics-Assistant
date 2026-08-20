import pytest

from src.insight_engine import generate_insights


def test_group_difference_generates_insight():

    analytics = {
        "group_comparisons": [
            {
                "category": "department",
                "metric": "salary",
                "highest_group": "IT",
                "highest_value": 87500,
                "lowest_group": "HR",
                "lowest_value": 51000,
                "percentage_gap": 71.57,
                "significance": "high",
            }
        ]
    }

    result = generate_insights(
        analytics
    )

    assert result["insight_count"] == 1

    insight = result["insights"][0]

    assert insight["type"] == "group_difference"
    assert insight["priority"] == "high"
    assert insight["evidence"]["highest_group"] == "IT"


def test_strong_correlation_generates_insight():

    analytics = {
        "correlations": {
            "study_hours": {
                "study_hours": 1.0,
                "score": 0.91,
            },
            "score": {
                "study_hours": 0.91,
                "score": 1.0,
            },
        }
    }

    result = generate_insights(
        analytics
    )

    assert result["insight_count"] == 1

    insight = result["insights"][0]

    assert insight["type"] == "correlation"
    assert insight["priority"] == "high"


def test_weak_correlation_is_ignored():

    analytics = {
        "correlations": {
            "age": {
                "age": 1.0,
                "salary": 0.32,
            },
            "salary": {
                "age": 0.32,
                "salary": 1.0,
            },
        }
    }

    result = generate_insights(
        analytics
    )

    assert result["insight_count"] == 0


def test_category_concentration_generates_insight():

    analytics = {
        "categorical_summary": {
            "department": {
                "unique_values": 4,
                "top_value": "IT",
                "top_value_count": 80,
                "top_value_percentage": 80.0,
            }
        }
    }

    result = generate_insights(
        analytics
    )

    assert result["insight_count"] == 1

    insight = result["insights"][0]

    assert insight["type"] == "category_concentration"
    assert insight["priority"] == "high"


def test_missing_values_generate_data_quality_insight():

    analytics = {
        "data_quality": {
            "missing_values": {
                "salary": 5,
            },
            "duplicate_rows": 0,
        }
    }

    result = generate_insights(
        analytics
    )

    assert result["insight_count"] == 1

    insight = result["insights"][0]

    assert insight["type"] == "data_quality"


def test_duplicate_rows_generate_insight():

    analytics = {
        "data_quality": {
            "missing_values": {},
            "duplicate_rows": 4,
        }
    }

    result = generate_insights(
        analytics
    )

    assert result["insight_count"] == 1

    assert (
        result["insights"][0]["type"]
        == "data_quality"
    )


def test_insights_are_ranked_by_priority():

    analytics = {
        "categorical_summary": {
            "department": {
                "unique_values": 3,
                "top_value": "IT",
                "top_value_count": 60,
                "top_value_percentage": 60.0,
            }
        },
        "correlations": {
            "age": {
                "age": 1.0,
                "salary": 0.95,
            },
            "salary": {
                "age": 0.95,
                "salary": 1.0,
            },
        },
    }

    result = generate_insights(
        analytics
    )

    assert result["insights"][0]["priority"] == "high"


def test_empty_analytics_returns_no_insights():

    result = generate_insights({})

    assert result["insights"] == []
    assert result["insight_count"] == 0


def test_invalid_input_is_rejected():

    with pytest.raises(TypeError):
        generate_insights([])


def test_multiple_insight_types_can_be_generated():

    analytics = {
        "group_comparisons": [
            {
                "category": "department",
                "metric": "salary",
                "highest_group": "IT",
                "highest_value": 90000,
                "lowest_group": "HR",
                "lowest_value": 50000,
                "percentage_gap": 80.0,
                "significance": "high",
            }
        ],
        "correlations": {
            "age": {
                "age": 1.0,
                "salary": 0.91,
            },
            "salary": {
                "age": 0.91,
                "salary": 1.0,
            },
        },
        "categorical_summary": {
            "department": {
                "unique_values": 3,
                "top_value": "IT",
                "top_value_count": 80,
                "top_value_percentage": 80.0,
            }
        },
        "data_quality": {
            "missing_values": {
                "salary": 2,
            },
            "duplicate_rows": 1,
        },
    }

    result = generate_insights(
        analytics
    )

    assert result["insight_count"] >= 4

    insight_types = {
        insight["type"]
        for insight in result["insights"]
    }

    assert "group_difference" in insight_types
    assert "correlation" in insight_types
    assert "category_concentration" in insight_types
    assert "data_quality" in insight_types