from typing import Any

import pandas as pd


SUPPORTED_ANALYSES = {
    "descriptive_analysis",
    "group_comparison",
    "correlation_analysis",
    "risk_analysis",
    "time_analysis",
    "trend_analysis",
    "data_quality_analysis",
}


# =========================================================
# HELPERS
# =========================================================

def _clean_list(
    value: Any,
) -> list[str]:
    """Convert a value into a clean list of strings."""

    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        value = [value]

    if not isinstance(
        value,
        (list, tuple),
    ):
        value = [value]

    return [
        str(item).strip()
        for item in value
        if item is not None
        and str(item).strip()
    ]


def _validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the DataFrame."""

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot execute analysis on an empty DataFrame."
        )


def _validate_metric(
    dataframe: pd.DataFrame,
    metric: str | None,
) -> None:
    """Validate that a metric exists and is numeric."""

    if not metric:
        raise ValueError(
            "Analysis requires a metric."
        )

    if metric not in dataframe.columns:
        raise ValueError(
            f"Metric column '{metric}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        dataframe[metric]
    ):
        raise ValueError(
            f"Metric column '{metric}' must be numeric."
        )


def _validate_dimension(
    dataframe: pd.DataFrame,
    dimension: str | None,
) -> None:
    """Validate that a dimension exists."""

    if not dimension:
        raise ValueError(
            "Analysis requires a dimension."
        )

    if dimension not in dataframe.columns:
        raise ValueError(
            f"Dimension column '{dimension}' does not exist."
        )


# =========================================================
# BUILD EXECUTION PLAN
# =========================================================

def build_execution_plan(
    decision: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a business decision into an execution plan.

    Important compatibility behavior:
        - trend_analysis stays trend_analysis
        - time_analysis stays time_analysis
    """

    if not isinstance(
        decision,
        dict,
    ):
        raise TypeError(
            "decision must be a dictionary."
        )

    analysis_type = decision.get(
        "recommended_analysis"
    )

    if not analysis_type:
        raise ValueError(
            "Decision must contain recommended_analysis."
        )

    analysis_type = str(
        analysis_type
    ).strip().lower()

    # ---------------------------------------------------------
    # Only normalize aliases that are genuinely aliases.
    #
    # Do NOT convert trend_analysis -> time_analysis because
    # existing tests explicitly distinguish them.
    # ---------------------------------------------------------

    aliases = {
        "descriptive": "descriptive_analysis",
        "group_difference": "group_comparison",
        "group_comparison_analysis": "group_comparison",
        "correlation": "correlation_analysis",
        "trend": "time_analysis",
        "time_series": "time_analysis",
        "risk": "risk_analysis",
        "data_quality": "data_quality_analysis",
    }

    analysis_type = aliases.get(
        analysis_type,
        analysis_type,
    )

    if analysis_type not in SUPPORTED_ANALYSES:
        raise ValueError(
            f"Unsupported analysis type: {analysis_type}"
        )

    metrics = _clean_list(
        decision.get(
            "metrics",
            [],
        )
    )

    dimensions = _clean_list(
        decision.get(
            "dimensions",
            [],
        )
    )

    return {
        "status": "planned",
        "analysis_type": analysis_type,
        "metric": (
            metrics[0]
            if metrics
            else None
        ),
        "metrics": metrics,
        "dimension": (
            dimensions[0]
            if dimensions
            else None
        ),
        "dimensions": dimensions,
        "decision": decision.get(
            "decision"
        ),
        "source_type": decision.get(
            "source_type"
        ),
    }


# =========================================================
# DESCRIPTIVE ANALYSIS
# =========================================================

def _execute_descriptive_analysis(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Execute basic descriptive analysis."""

    numeric = dataframe.select_dtypes(
        include="number"
    )

    summary = {}

    if not numeric.empty:
        summary = numeric.describe().to_dict()

    return {
        "analysis_type": "descriptive_analysis",
        "status": "completed",
        "rows_analyzed": len(dataframe),
        "columns_analyzed": len(dataframe.columns),
        "numeric_summary": summary,
    }


# =========================================================
# GROUP COMPARISON
# =========================================================

def _execute_group_comparison(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare a numeric metric across groups.

    Rules:
    - metric must be explicitly supplied;
    - group/dimension may be supplied or inferred from the
      first suitable categorical column.
    """

    # ---------------------------------------------------------
    # Metric
    # ---------------------------------------------------------

    metric = plan.get(
        "metric"
    )

    if not metric:
        metrics = _clean_list(
            plan.get(
                "metrics",
                [],
            )
        )

        if metrics:
            metric = metrics[0]

    # Metric remains mandatory.
    _validate_metric(
        dataframe,
        metric,
    )

    # ---------------------------------------------------------
    # Group / dimension
    # ---------------------------------------------------------

    dimension = plan.get(
        "dimension"
    )

    if not dimension:
        dimensions = _clean_list(
            plan.get(
                "dimensions",
                [],
            )
        )

        if dimensions:
            dimension = dimensions[0]

    # ---------------------------------------------------------
    # Infer the group column when the plan does not contain one.
    #
    # This is required by the existing execution test:
    #
    # category | revenue
    #
    # where category must be detected automatically.
    # ---------------------------------------------------------

    if not dimension:

        for column in dataframe.columns:

            if column == metric:
                continue

            series = dataframe[column]

            # Skip datetime columns.
            if pd.api.types.is_datetime64_any_dtype(
                series
            ):
                continue

            # Skip numeric columns.
            if pd.api.types.is_numeric_dtype(
                series
            ):
                continue

            # Suitable categorical/object/string column.
            if (
                pd.api.types.is_object_dtype(
                    series
                )
                or pd.api.types.is_string_dtype(
                    series
                )
                or isinstance(
                    series.dtype,
                    pd.CategoricalDtype,
                )
            ):

                if series.dropna().nunique() >= 2:
                    dimension = str(column)
                    break

    if not dimension:
        raise ValueError(
            "Group comparison requires a group column."
        )

    _validate_dimension(
        dataframe,
        dimension,
    )

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    working = dataframe[
        [dimension, metric]
    ].copy()

    working[metric] = pd.to_numeric(
        working[metric],
        errors="coerce",
    )

    working = working.dropna(
        subset=[
            dimension,
            metric,
        ]
    )

    if working.empty:
        raise ValueError(
            "No valid data available for group comparison."
        )

    # ---------------------------------------------------------
    # Aggregate by SUM
    #
    # Electronics = 1000 + 1500 = 2500
    # Furniture   = 500 + 700 = 1200
    # ---------------------------------------------------------

    grouped = (
        working
        .groupby(
            dimension,
            dropna=False,
        )[metric]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if grouped.empty:
        raise ValueError(
            "No groups available for comparison."
        )

    # ---------------------------------------------------------
    # Highest / lowest
    # ---------------------------------------------------------

    highest_group = grouped.idxmax()
    lowest_group = grouped.idxmin()

    highest_value = float(
        grouped.loc[
            highest_group
        ]
    )

    lowest_value = float(
        grouped.loc[
            lowest_group
        ]
    )

    # ---------------------------------------------------------
    # Percentage difference
    # ---------------------------------------------------------

    if lowest_value == 0:

        if highest_value == 0:
            percentage_gap = 0.0
        else:
            percentage_gap = 100.0

    else:

        percentage_gap = (
            abs(
                highest_value
                - lowest_value
            )
            / abs(lowest_value)
            * 100
        )

    # ---------------------------------------------------------
    # Significance
    # ---------------------------------------------------------

    if percentage_gap >= 50:
        significance = "high"

    elif percentage_gap >= 20:
        significance = "medium"

    else:
        significance = "low"

    # ---------------------------------------------------------
    # Legacy-compatible group_values
    # ---------------------------------------------------------

    group_values = {
        str(group): float(value)
        for group, value in grouped.items()
    }

    return {
        "analysis_type": "group_comparison",
        "status": "completed",

        "metric": metric,

        "group_column": dimension,
        "dimension": dimension,

        "highest_group": highest_group,
        "lowest_group": lowest_group,

        "highest_value": highest_value,
        "lowest_value": lowest_value,

        "percentage_gap": float(
            percentage_gap
        ),

        "significance": significance,

        "group_values": group_values,

        "groups": group_values,
    }

# =========================================================
# CORRELATION ANALYSIS
# =========================================================

def _execute_correlation(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute correlation analysis.

    Requires at least two explicitly supplied metrics.
    """

    metrics = _clean_list(
        plan.get(
            "metrics",
            [],
        )
    )

    if len(metrics) < 2:
        raise ValueError(
            "Correlation analysis requires at least two metrics."
        )

    for metric in metrics:
        _validate_metric(
            dataframe,
            metric,
        )

    correlation_matrix = (
        dataframe[metrics]
        .corr()
    )

    first_metric = metrics[0]
    second_metric = metrics[1]

    correlation_value = float(
        correlation_matrix.loc[
            first_metric,
            second_metric,
        ]
    )

    return {
        "analysis_type": "correlation_analysis",
        "status": "completed",
        "metrics": metrics,

        # Required by existing tests.
        "correlation": correlation_value,

        "correlation_matrix": (
            correlation_matrix.to_dict()
        ),
    }


# =========================================================
# TIME / TREND ANALYSIS
# =========================================================

def _find_time_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Find the first datetime column."""

    for column in dataframe.columns:

        if pd.api.types.is_datetime64_any_dtype(
            dataframe[column]
        ):
            return str(column)

    return None


def _execute_time_or_trend(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute time/trend analysis.

    The calculation is shared, but the returned
    analysis_type preserves the caller's requested name.
    """

    metric = plan.get(
        "metric"
    )

    if not metric:

        metrics = _clean_list(
            plan.get(
                "metrics",
                [],
            )
        )

        if metrics:
            metric = metrics[0]

    # Existing tests require ValueError when metric is absent.
    _validate_metric(
        dataframe,
        metric,
    )

    time_column = _find_time_column(
        dataframe
    )

    if time_column is None:
        raise ValueError(
            "Trend analysis requires a datetime column."
        )

    working = dataframe[
        [time_column, metric]
    ].copy()

    working = working.dropna(
        subset=[
            time_column,
            metric,
        ]
    )

    if working.empty:
        raise ValueError(
            "No valid data available for trend analysis."
        )

    working = working.sort_values(
        by=time_column
    )

    values = (
        working[metric]
        .tolist()
    )

    if len(values) < 2:
        raise ValueError(
            "Trend analysis requires at least two observations."
        )

    first_value = float(
        values[0]
    )

    last_value = float(
        values[-1]
    )

    if last_value > first_value:
        direction = "increasing"

    elif last_value < first_value:
        direction = "decreasing"

    else:
        direction = "stable"

    requested_type = str(
        plan.get(
            "analysis_type",
            "trend_analysis",
        )
    ).strip().lower()

    if requested_type not in {
        "trend_analysis",
        "time_analysis",
    }:
        requested_type = "trend_analysis"

    return {
        "analysis_type": requested_type,
        "status": "completed",
        "metric": metric,
        "time_column": time_column,

        # Required by existing tests.
        "direction": direction,
        "first_value": first_value,
        "last_value": last_value,
        "change": last_value - first_value,
    }


# =========================================================
# RISK ANALYSIS
# =========================================================

def _execute_risk(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute concentration-oriented risk analysis.

    Metric is intentionally required.
    """

    metric = plan.get(
        "metric"
    )

    if not metric:

        metrics = _clean_list(
            plan.get(
                "metrics",
                [],
            )
        )

        if metrics:
            metric = metrics[0]

    _validate_metric(
        dataframe,
        metric,
    )

    values = (
        pd.to_numeric(
            dataframe[metric],
            errors="coerce",
        )
        .dropna()
    )

    if values.empty:
        raise ValueError(
            "Risk analysis requires at least one numeric value."
        )

    total = float(
        values.sum()
    )

    highest_value = float(
        values.max()
    )

    if total == 0:
        highest_share = 0.0

    else:
        highest_share = (
            highest_value
            / total
        )

    if highest_share >= 0.70:
        risk_level = "high"

    elif highest_share >= 0.40:
        risk_level = "medium"

    else:
        risk_level = "low"

    return {
        "analysis_type": "risk_analysis",
        "status": "completed",
        "metric": metric,

        # Required fields.
        "total": total,
        "highest_value": highest_value,
        "highest_share": highest_share,
        "risk_level": risk_level,

        "mean": float(
            values.mean()
        ),
    }


# =========================================================
# DATA QUALITY
# =========================================================

def _execute_data_quality(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Execute basic data-quality checks."""

    missing_values = {
        str(column): int(
            dataframe[column].isna().sum()
        )
        for column in dataframe.columns
        if dataframe[column].isna().sum() > 0
    }

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    return {
        "analysis_type": "data_quality_analysis",
        "status": "completed",
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
    }


# =========================================================
# PUBLIC EXECUTION API
# =========================================================

def execute_analysis(
    dataframe: pd.DataFrame,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a concrete analysis plan.
    """

    _validate_dataframe(
        dataframe
    )

    if not isinstance(
        plan,
        dict,
    ):
        raise TypeError(
            "plan must be a dictionary."
        )

    analysis_type = plan.get(
        "analysis_type"
    )

    if not analysis_type:
        raise ValueError(
            "Plan must contain analysis_type."
        )

    analysis_type = str(
        analysis_type
    ).strip().lower()

    if analysis_type == "descriptive_analysis":
        return _execute_descriptive_analysis(
            dataframe,
            plan,
        )

    if analysis_type == "group_comparison":
        return _execute_group_comparison(
            dataframe,
            plan,
        )

    if analysis_type == "correlation_analysis":
        return _execute_correlation(
            dataframe,
            plan,
        )

    if analysis_type in {
        "trend_analysis",
        "time_analysis",
    }:
        return _execute_time_or_trend(
            dataframe,
            plan,
        )

    if analysis_type == "risk_analysis":
        return _execute_risk(
            dataframe,
            plan,
        )

    if analysis_type == "data_quality_analysis":
        return _execute_data_quality(
            dataframe,
            plan,
        )

    raise ValueError(
        f"Unsupported analysis type: {analysis_type}"
    )