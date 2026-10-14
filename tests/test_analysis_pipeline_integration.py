import pandas as pd

from src.analysis_execution import (
    build_execution_plan,
    execute_analysis,
)


def test_group_comparison_pipeline():

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

    decision = {
        "decision": (
            "Compare revenue performance across categories."
        ),
        "recommended_analysis": "group_comparison",
        "evidence": [
            "Revenue differs between categories."
        ],
        "metrics": [
            "revenue"
        ],
        "type": "group_comparison",
    }

    plan = build_execution_plan(decision)

    assert plan["analysis_type"] == "group_comparison"
    assert plan["metric"] == "revenue"
    assert plan["status"] == "planned"

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