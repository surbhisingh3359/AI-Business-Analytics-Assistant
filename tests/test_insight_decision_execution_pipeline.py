import pandas as pd

from src.insight_ranker import rank_insights
from src.decision_engine import build_decision_layer
from src.analysis_execution import (
    build_execution_plan,
    execute_analysis,
)


def test_insight_to_decision_to_execution_pipeline():

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

    # ---------------------------------------------------------
    # 1. Simulated analytical insights
    # ---------------------------------------------------------

    insights = [
        {
            "finding": (
                "Revenue is higher for Electronics "
                "than Furniture."
            ),
            "type": "group_comparison",
            "priority": "high",
            "confidence": "high",
            "metric": "revenue",
            "category": "category",
        },
        {
            "finding": (
                "Revenue is higher for Electronics "
                "than Furniture."
            ),
            "type": "group_comparison",
            "priority": "medium",
            "confidence": "medium",
            "metric": "revenue",
            "category": "category",
        },
    ]

    # ---------------------------------------------------------
    # 2. Rank insights
    # ---------------------------------------------------------

    ranked = rank_insights(
        insights,
        max_insights=5,
    )

    assert ranked
    assert ranked[0]["type"] == "group_comparison"

    # Exact duplicate should be removed.
    assert len(ranked) == 1

    # Highest-priority insight should be retained.
    assert ranked[0]["priority"] == "high"

    # ---------------------------------------------------------
    # 3. Convert insight into decision
    # ---------------------------------------------------------

    decisions = build_decision_layer(
        ranked,
        max_decisions=5,
    )

    assert len(decisions) == 1

    decision = decisions[0]

    assert decision["recommended_analysis"] == (
        "group_comparison"
    )

    assert decision["metrics"] == [
        "revenue"
    ]

    assert decision["dimensions"] == [
        "category"
    ]

    # ---------------------------------------------------------
    # 4. Build execution plan
    # ---------------------------------------------------------

    plan = build_execution_plan(
        decision
    )

    assert plan["analysis_type"] == (
        "group_comparison"
    )

    assert plan["metric"] == "revenue"

    assert plan["dimensions"] == [
        "category"
    ]

    assert plan["status"] == "planned"

    # ---------------------------------------------------------
    # 5. Execute analysis
    # ---------------------------------------------------------

    result = execute_analysis(
        dataframe,
        plan,
    )

    assert result["analysis_type"] == (
        "group_comparison"
    )

    assert result["status"] == "completed"

    assert result["metric"] == "revenue"

    assert result["group_column"] == "category"

    assert result["highest_group"] == (
        "Electronics"
    )

    assert result["lowest_group"] == (
        "Furniture"
    )

    assert result["group_values"]["Electronics"] == 2500

    assert result["group_values"]["Furniture"] == 1200