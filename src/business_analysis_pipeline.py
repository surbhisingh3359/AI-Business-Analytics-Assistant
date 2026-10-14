from typing import Any

import pandas as pd

from src.ai_context import build_ai_context
from src.analysis_execution import (
    build_execution_plan,
    execute_analysis,
)
from src.decision_engine import build_decision_layer
from src.predictive_analysis import (
    can_run_prediction,
    detect_prediction_targets,
    detect_prediction_task,
    run_prediction,
)

# =========================================================
# VALIDATION
# =========================================================

def _validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the input dataset before entering
    the analytical pipeline.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot analyze an empty dataset."
        )


def _validate_limits(
    max_insights: int,
    max_decisions: int,
) -> None:
    """Validate pipeline result limits."""

    if not isinstance(
        max_insights,
        int,
    ):
        raise TypeError(
            "max_insights must be an integer."
        )

    if max_insights <= 0:
        raise ValueError(
            "max_insights must be greater than zero."
        )

    if not isinstance(
        max_decisions,
        int,
    ):
        raise TypeError(
            "max_decisions must be an integer."
        )

    if max_decisions <= 0:
        raise ValueError(
            "max_decisions must be greater than zero."
        )


def _requires_metric(
    analysis_type: str,
) -> bool:
    """
    Determine whether an analysis requires metric
    information before execution.
    """

    return analysis_type in {
        "group_comparison",
        "correlation_analysis",
        "risk_analysis",
        "time_analysis",
        "trend_analysis",
    }


# =========================================================
# DECISION EXECUTION
# =========================================================

def _execute_decisions(
    dataframe: pd.DataFrame,
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert business decisions into execution plans
    and execute them.

    Possible statuses:

        completed
        skipped
        failed

    A skipped decision means the decision is valid,
    but there is not enough information to safely
    execute it.
    """

    execution_results: list[
        dict[str, Any]
    ] = []

    for decision in decisions:

        if not isinstance(
            decision,
            dict,
        ):
            continue

        recommended_analysis = str(
            decision.get(
                "recommended_analysis",
                "",
            )
        ).strip().lower()

        metrics = decision.get(
            "metrics",
            [],
        )

        if metrics is None:
            metrics = []

        if isinstance(
            metrics,
            str,
        ):
            metrics = [
                metrics
            ]

        # -----------------------------------------------------
        # Skip analyses that require a metric but don't have
        # one.
        # -----------------------------------------------------

        if (
            _requires_metric(
                recommended_analysis
            )
            and not metrics
        ):

            execution_results.append(
                {
                    "decision": decision,
                    "plan": None,
                    "result": None,
                    "status": "skipped",
                    "error": (
                        "This decision does not contain "
                        "enough metric information for "
                        "execution."
                    ),
                }
            )

            continue

        # -----------------------------------------------------
        # Build execution plan + execute
        # -----------------------------------------------------

        try:

            plan = build_execution_plan(
                decision
            )

            result = execute_analysis(
                dataframe,
                plan,
            )

            execution_results.append(
                {
                    "decision": decision,
                    "plan": plan,
                    "result": result,
                    "status": "completed",
                    "error": None,
                }
            )

        except (
            TypeError,
            ValueError,
            KeyError,
        ) as exc:

            execution_results.append(
                {
                    "decision": decision,
                    "plan": None,
                    "result": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return execution_results


def _split_execution_results(
    execution_results: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Split execution results into completed,
    failed and skipped collections.
    """

    completed_results = [
        item
        for item in execution_results
        if item.get(
            "status"
        ) == "completed"
    ]

    failed_results = [
        item
        for item in execution_results
        if item.get(
            "status"
        ) == "failed"
    ]

    skipped_results = [
        item
        for item in execution_results
        if item.get(
            "status"
        ) == "skipped"
    ]

    return (
        completed_results,
        failed_results,
        skipped_results,
    )


def _pipeline_status(
    failed_results: list[dict[str, Any]],
) -> str:
    """
    Determine overall pipeline status.

    Skipped analyses are not considered pipeline failures.
    """

    if failed_results:
        return "completed_with_errors"

    return "completed"


# =========================================================
# PREDICTIVE INFORMATION
# =========================================================

def _build_predictive_options(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build predictive options for the application.

    `available` means prediction targets were detected.

    `model_ready` means the dataset has enough usable
    observations to train and evaluate a model.
    """

    targets = detect_prediction_targets(
        dataframe
    )

    readiness = can_run_prediction(
        dataframe
    )

    regression_candidates = targets.get(
        "regression_candidates",
        [],
    )

    classification_candidates = targets.get(
        "classification_candidates",
        [],
    )

    has_targets = bool(
        regression_candidates
        or classification_candidates
    )

    return {
        "available": has_targets,

        "model_ready": readiness["available"],

        "usable_rows": readiness[
            "usable_rows"
        ],

        "minimum_rows": readiness[
            "minimum_rows"
        ],

        "reason": readiness[
            "reason"
        ],

        "regression_candidates": [
            str(column)
            for column in regression_candidates
        ],

        "classification_candidates": [
            str(column)
            for column in classification_candidates
        ],
    }

def run_business_prediction(
    dataframe: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    """
    Run predictive analysis for a user-selected target.

    This is the application-facing prediction API.

    The UI should call this function instead of directly
    interacting with scikit-learn.
    """

    _validate_dataframe(
        dataframe
    )

    if not isinstance(
        target,
        str,
    ):
        raise TypeError(
            "target must be a string."
        )

    if not target.strip():
        raise ValueError(
            "target cannot be empty."
        )

    task = detect_prediction_task(
        dataframe,
        target,
    )

    prediction = run_prediction(
        dataframe,
        target,
    )

    return {
        "status": prediction.get(
            "status",
            "completed",
        ),
        "task": task,
        "target": target,
        "model_name": prediction.get(
            "model_name"
        ),
        "feature_columns": prediction.get(
            "feature_columns",
            [],
        ),
        "numeric_features": prediction.get(
            "numeric_features",
            [],
        ),
        "categorical_features": prediction.get(
            "categorical_features",
            [],
        ),
        "train_rows": prediction.get(
            "train_rows",
            0,
        ),
        "test_rows": prediction.get(
            "test_rows",
            0,
        ),
        "metrics": prediction.get(
            "metrics",
            {},
        ),
        "predictions": prediction.get(
            "predictions",
            [],
        ),
        "actual_values": prediction.get(
            "actual_values",
            [],
        ),
    }


# =========================================================
# MAIN BUSINESS ANALYSIS PIPELINE
# =========================================================

def run_business_analysis(
    dataframe: pd.DataFrame,
    max_insights: int = 5,
    max_decisions: int = 5,
) -> dict[str, Any]:
    """
    Run the deterministic business-analysis pipeline.

    Pipeline:

        DataFrame
            ↓
        AI Context
            ↓
        Semantic Roles
            ↓
        Analysis Plan
            ↓
        Universal Analytics
            ↓
        Insight Generation
            ↓
        Insight Ranking
            ↓
        Decision Layer
            ↓
        Execution Plans
            ↓
        Analysis Execution

    Predictive target detection is also included in the
    returned result, but model training is intentionally
    triggered separately through run_business_prediction().
    """

    # ---------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------

    _validate_dataframe(
        dataframe
    )

    _validate_limits(
        max_insights=max_insights,
        max_decisions=max_decisions,
    )

    # ---------------------------------------------------------
    # 2. Build analytical context
    # ---------------------------------------------------------

    context = build_ai_context(
        dataframe
    )

    if not isinstance(
        context,
        dict,
    ):
        raise TypeError(
            "build_ai_context must return a dictionary."
        )

    # ---------------------------------------------------------
    # 3. Extract ranked insights
    # ---------------------------------------------------------

    ranked_insights = context.get(
        "insights",
        [],
    )

    if not isinstance(
        ranked_insights,
        list,
    ):
        ranked_insights = []

    ranked_insights = (
        ranked_insights[
            :max_insights
        ]
    )

    # ---------------------------------------------------------
    # 4. Convert insights into decisions
    # ---------------------------------------------------------

    decisions = build_decision_layer(
        ranked_insights,
        max_decisions=max_decisions,
    )

    # ---------------------------------------------------------
    # 5. Execute recommended decisions
    # ---------------------------------------------------------

    execution_results = _execute_decisions(
        dataframe,
        decisions,
    )

    # ---------------------------------------------------------
    # 6. Split execution results
    # ---------------------------------------------------------

    (
        completed_results,
        failed_results,
        skipped_results,
    ) = _split_execution_results(
        execution_results
    )

    # ---------------------------------------------------------
    # 7. Build predictive options
    # ---------------------------------------------------------

    predictive_options = (
        _build_predictive_options(
            dataframe
        )
    )

    # ---------------------------------------------------------
    # 8. Overall status
    # ---------------------------------------------------------

    status = _pipeline_status(
        failed_results
    )

    # ---------------------------------------------------------
    # 9. Return complete application result
    # ---------------------------------------------------------

    return {
        "status": status,

        # -----------------------------------------------------
        # Dataset
        # -----------------------------------------------------

        "data": dataframe,

        "dataset": context.get(
            "dataset",
            {},
        ),

        # -----------------------------------------------------
        # Semantic understanding
        # -----------------------------------------------------

        "semantic_roles": context.get(
            "semantic_roles",
            {},
        ),

        # -----------------------------------------------------
        # Analysis planning
        # -----------------------------------------------------

        "analysis_plan": context.get(
            "analysis_plan",
            {},
        ),

        # -----------------------------------------------------
        # Universal analytics
        # -----------------------------------------------------

        "analytics": context.get(
            "analytics",
            {},
        ),

        # -----------------------------------------------------
        # Descriptive insights
        # -----------------------------------------------------

        "insights": ranked_insights,

        "insight_count": len(
            ranked_insights
        ),

        # -----------------------------------------------------
        # Business decisions
        # -----------------------------------------------------

        "decisions": decisions,

        "decision_count": len(
            decisions
        ),

        # -----------------------------------------------------
        # Analytical execution
        # -----------------------------------------------------

        "execution_results": (
            execution_results
        ),

        "completed_executions": (
            completed_results
        ),

        "failed_executions": (
            failed_results
        ),

        "skipped_executions": (
            skipped_results
        ),

        "execution_count": len(
            execution_results
        ),

        # -----------------------------------------------------
        # Predictive analytics
        # -----------------------------------------------------

        "predictive_options": (
            predictive_options
        ),

        "predictive_result": None,

        # -----------------------------------------------------
        # Business questions
        # -----------------------------------------------------

        "business_questions": context.get(
            "business_questions",
            [],
        ),

        "business_question_count": context.get(
            "business_question_count",
            0,
        ),
    }


# =========================================================
# CSV ENTRY POINT
# =========================================================

def run_business_analysis_from_csv(
    file_path: str,
    max_insights: int = 5,
    max_decisions: int = 5,
) -> dict[str, Any]:
    """
    Load a CSV file and run the deterministic
    business-analysis pipeline.
    """

    if not isinstance(
        file_path,
        str,
    ):
        raise TypeError(
            "file_path must be a string."
        )

    if not file_path.strip():
        raise ValueError(
            "file_path cannot be empty."
        )

    dataframe = pd.read_csv(
        file_path
    )

    return run_business_analysis(
        dataframe,
        max_insights=max_insights,
        max_decisions=max_decisions,
    )