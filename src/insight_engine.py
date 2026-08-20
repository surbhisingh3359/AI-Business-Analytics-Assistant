from typing import Any

import pandas as pd


def _make_insight(
    insight_type: str,
    title: str,
    finding: str,
    evidence: dict[str, Any],
    priority: str = "medium",
    confidence: str = "high",
) -> dict[str, Any]:
    """Create a standardized insight."""

    return {
        "type": insight_type,
        "priority": priority,
        "title": title,
        "finding": finding,
        "evidence": evidence,
        "confidence": confidence,
    }


def _extract_group_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract meaningful insights from group comparisons."""

    insights = []

    comparisons = analytics.get(
        "group_comparisons",
        [],
    )

    if not isinstance(comparisons, list):
        return insights

    for comparison in comparisons:

        category = comparison.get("category")
        metric = comparison.get("metric")
        highest_group = comparison.get("highest_group")
        lowest_group = comparison.get("lowest_group")
        highest_value = comparison.get("highest_value")
        lowest_value = comparison.get("lowest_value")
        percentage_gap = comparison.get("percentage_gap")

        if category is None or metric is None:
            continue

        significance = comparison.get(
            "significance",
            "medium",
        )

        priority = (
            "high"
            if significance == "high"
            else "medium"
        )

        insights.append(
            _make_insight(
                insight_type="group_difference",
                priority=priority,
                title=(
                    f"{highest_group} has the highest "
                    f"average {metric}"
                ),
                finding=(
                    f"{highest_group} has the highest average "
                    f"{metric}, while {lowest_group} has the "
                    f"lowest. The difference is approximately "
                    f"{percentage_gap:.2f}%."
                ),
                evidence={
                    "category": category,
                    "metric": metric,
                    "highest_group": highest_group,
                    "highest_value": highest_value,
                    "lowest_group": lowest_group,
                    "lowest_value": lowest_value,
                    "percentage_gap": percentage_gap,
                },
            )
        )

    return insights


def _extract_correlation_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract strong relationships between numeric variables."""

    insights = []

    correlations = analytics.get(
        "correlations",
        {},
    )

    if not isinstance(correlations, dict):
        return insights

    seen_pairs: set[tuple[str, str]] = set()

    for column, related_columns in correlations.items():

        if not isinstance(related_columns, dict):
            continue

        for other_column, value in related_columns.items():

            if column == other_column:
                continue

            if not isinstance(value, (int, float)):
                continue

            pair = tuple(
                sorted(
                    [str(column), str(other_column)]
                )
            )

            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            absolute_correlation = abs(float(value))

            if absolute_correlation < 0.70:
                continue

            if absolute_correlation >= 0.90:
                priority = "high"
                confidence = "high"
            else:
                priority = "medium"
                confidence = "medium"

            direction = (
                "positive"
                if value > 0
                else "negative"
            )

            insights.append(
                _make_insight(
                    insight_type="correlation",
                    priority=priority,
                    confidence=confidence,
                    title=(
                        f"Strong {direction} relationship between "
                        f"{column} and {other_column}"
                    ),
                    finding=(
                        f"{column} and {other_column} have a "
                        f"{direction} correlation of "
                        f"{value:.2f}."
                    ),
                    evidence={
                        "column_1": column,
                        "column_2": other_column,
                        "correlation": float(value),
                    },
                )
            )

    return insights


def _extract_categorical_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Identify dominant categorical values."""

    insights = []

    summaries = analytics.get(
        "categorical_summary",
        {},
    )

    if not isinstance(summaries, dict):
        return insights

    for column, summary in summaries.items():

        if not isinstance(summary, dict):
            continue

        top_value = summary.get("top_value")
        top_percentage = summary.get(
            "top_value_percentage"
        )
        unique_values = summary.get(
            "unique_values"
        )

        if (
            top_value is None
            or top_percentage is None
        ):
            continue

        # Only report meaningful concentration.
        if top_percentage < 50:
            continue

        priority = (
            "high"
            if top_percentage >= 75
            else "medium"
        )

        insights.append(
            _make_insight(
                insight_type="category_concentration",
                priority=priority,
                title=(
                    f"{column} is concentrated in "
                    f"{top_value}"
                ),
                finding=(
                    f"The value '{top_value}' represents "
                    f"{top_percentage:.2f}% of observations "
                    f"in {column}."
                ),
                evidence={
                    "column": column,
                    "top_value": top_value,
                    "top_value_percentage": float(
                        top_percentage
                    ),
                    "unique_values": unique_values,
                },
            )
        )

    return insights


def _extract_data_quality_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Identify important data-quality problems."""

    insights = []

    quality = analytics.get(
        "data_quality",
        {},
    )

    if not isinstance(quality, dict):
        return insights

    missing_values = quality.get(
        "missing_values",
        {},
    )

    if isinstance(missing_values, dict):

        for column, missing_count in missing_values.items():

            if not isinstance(
                missing_count,
                (int, float),
            ):
                continue

            if missing_count <= 0:
                continue

            insights.append(
                _make_insight(
                    insight_type="data_quality",
                    priority="medium",
                    confidence="high",
                    title=(
                        f"Missing values detected in {column}"
                    ),
                    finding=(
                        f"{column} contains "
                        f"{int(missing_count)} missing "
                        f"values that may affect analysis."
                    ),
                    evidence={
                        "column": column,
                        "missing_count": int(
                            missing_count
                        ),
                    },
                )
            )

    duplicate_rows = quality.get(
        "duplicate_rows",
        0,
    )

    if (
        isinstance(duplicate_rows, (int, float))
        and duplicate_rows > 0
    ):
        insights.append(
            _make_insight(
                insight_type="data_quality",
                priority="medium",
                confidence="high",
                title="Duplicate rows detected",
                finding=(
                    f"The dataset contains "
                    f"{int(duplicate_rows)} duplicate rows."
                ),
                evidence={
                    "duplicate_rows": int(
                        duplicate_rows
                    )
                },
            )
        )

    return insights


def generate_insights(
    analytics: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate meaningful, dataset-agnostic insights
    from universal analytics results.
    """

    if not isinstance(analytics, dict):
        raise TypeError(
            "analytics must be a dictionary."
        )

    insights: list[dict[str, Any]] = []

    insights.extend(
        _extract_group_insights(
            analytics
        )
    )

    insights.extend(
        _extract_correlation_insights(
            analytics
        )
    )

    insights.extend(
        _extract_categorical_insights(
            analytics
        )
    )

    insights.extend(
        _extract_data_quality_insights(
            analytics
        )
    )

    priority_order = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    insights.sort(
        key=lambda item: priority_order.get(
            item.get("priority", "medium"),
            1,
        )
    )

    return {
        "insights": insights,
        "insight_count": len(insights),
    }