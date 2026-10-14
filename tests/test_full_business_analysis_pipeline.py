import pandas as pd

from src.analysis_planner import build_analysis_plan
from src.insight_engine import generate_insights
from src.insight_ranker import rank_insights
from src.decision_engine import build_decision_layer
from src.analysis_execution import (
    build_execution_plan,
    execute_analysis,
)


def test_full_business_analysis_pipeline():

    # ---------------------------------------------------------
    # 1. Create sample business dataset
    # ---------------------------------------------------------

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
    # 2. Define semantic roles
    # ---------------------------------------------------------

    semantic_roles = {
        "category": "categorical",
        "revenue": "monetary",
    }

    # ---------------------------------------------------------
    # 3. Build analysis plan
    # ---------------------------------------------------------

    analysis_plan = build_analysis_plan(
        dataframe,
        semantic_roles,
    )

    assert isinstance(
        analysis_plan,
        dict,
    )

    assert "analyses" in analysis_plan
    assert "columns_by_role" in analysis_plan

    assert (
        "group_comparison_analysis"
        in analysis_plan["analyses"]
    )

    # ---------------------------------------------------------
    # 4. Simulate analysis execution output
    # ---------------------------------------------------------

    analysis_results = {
        "dataset_overview": {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
        },
        "group_comparisons": [
            {
                "category": "category",
                "metric": "revenue",
                "group_count": 2,
                "highest_group": "Electronics",
                "highest_value": 1250.0,
                "lowest_group": "Furniture",
                "lowest_value": 600.0,
                "percentage_gap": (
                    (1250.0 - 600.0)
                    / 600.0
                    * 100
                ),
                "significance": "high",
            }
        ],
    }

    # ---------------------------------------------------------
    # 5. Generate insights
    # ---------------------------------------------------------

    insight_result = generate_insights(
        analysis_results
    )

    assert isinstance(
        insight_result,
        dict,
    )

    assert "insights" in insight_result
    assert "insight_count" in insight_result

    assert (
        insight_result["insight_count"]
        > 0
    )

    insights = insight_result["insights"]

    assert isinstance(
        insights,
        list,
    )

    assert len(insights) > 0

    # ---------------------------------------------------------
    # 6. Rank insights
    # ---------------------------------------------------------

    ranked_insights = rank_insights(
        insights,
        max_insights=5,
    )

    assert isinstance(
        ranked_insights,
        list,
    )

    assert len(ranked_insights) > 0

    # Highest-priority insight should be first.
    assert ranked_insights[0]["priority"] == "high"

    # ---------------------------------------------------------
    # 7. Convert insights into decisions
    # ---------------------------------------------------------

    decisions = build_decision_layer(
        ranked_insights,
        max_decisions=5,
    )

    assert isinstance(
        decisions,
        list,
    )

    assert len(decisions) > 0

    decision = decisions[0]

    assert "decision" in decision
    assert "recommended_analysis" in decision
    assert "metrics" in decision

    # ---------------------------------------------------------
    # 8. Build execution plan
    # ---------------------------------------------------------

    execution_plan = build_execution_plan(
        decision
    )

    assert isinstance(
        execution_plan,
        dict,
    )

    assert (
        execution_plan["status"]
        == "planned"
    )

    assert (
        execution_plan["analysis_type"]
        == "group_comparison"
    )

    # ---------------------------------------------------------
    # 9. Execute analysis
    # ---------------------------------------------------------

    execution_result = execute_analysis(
        dataframe,
        execution_plan,
    )

    assert isinstance(
        execution_result,
        dict,
    )

    assert (
        execution_result["status"]
        == "completed"
    )

    assert (
        execution_result["analysis_type"]
        == "group_comparison"
    )

    assert (
        execution_result["metric"]
        == "revenue"
    )

    # ---------------------------------------------------------
    # 10. Validate business result
    # ---------------------------------------------------------

    assert (
        execution_result["highest_group"]
        == "Electronics"
    )

    assert (
        execution_result["lowest_group"]
        == "Furniture"
    )

    assert (
        execution_result["group_values"]["Electronics"]
        == 2500
    )

    assert (
        execution_result["group_values"]["Furniture"]
        == 1200
    )