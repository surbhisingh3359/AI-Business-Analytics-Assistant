from typing import Any

import pandas as pd

from src.significance import describe_verdict


def _priority_from_confidence(
    confidence: str,
) -> str:
    """
    Derive reporting priority from statistical confidence.

    Priority used to follow the size of the gap, which promoted findings
    that no test would support. It now follows the confidence grade, so an
    untestable or noisy result is never surfaced at the top.
    """

    if confidence == "high":
        return "high"

    if confidence == "medium":
        return "medium"

    return "low"


def _make_insight(
    insight_type: str,
    title: str,
    finding: str,
    evidence: dict[str, Any],
    priority: str = "medium",
    confidence: str = "high",
    metric: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """
    Create a standardized insight.

    Analytical metadata such as metric/category is kept both
    at the top level and inside evidence so downstream layers
    can consume the insight without parsing nested structures.
    """

    result = {
        "type": insight_type,
        "priority": priority,
        "title": title,
        "finding": finding,
        "evidence": evidence,
        "confidence": confidence,
    }

    if metric is not None:
        result["metric"] = str(metric)

    if category is not None:
        result["category"] = str(category)

    return result


def _extract_group_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract meaningful insights from group comparisons."""

    insights = []

    comparisons = analytics.get(
        "group_comparisons",
        [],
    )

    if not isinstance(
        comparisons,
        list,
    ):
        return insights

    for comparison in comparisons:

        category = comparison.get(
            "category"
        )

        metric = comparison.get(
            "metric"
        )

        highest_group = comparison.get(
            "highest_group"
        )

        lowest_group = comparison.get(
            "lowest_group"
        )

        highest_value = comparison.get(
            "highest_value"
        )

        lowest_value = comparison.get(
            "lowest_value"
        )

        percentage_gap = comparison.get(
            "percentage_gap"
        )

        if category is None or metric is None:
            continue

        statistics = comparison.get("statistics")

        if not isinstance(statistics, dict):
            statistics = {}

        # Confidence and priority both come from the test now, so a large
        # gap measured on a handful of rows can no longer be promoted.
        confidence = str(
            statistics.get(
                "confidence",
                comparison.get("significance", "medium"),
            )
        )

        priority = _priority_from_confidence(confidence)

        support = describe_verdict(statistics)

        finding = (
            f"{highest_group} has the highest average "
            f"{metric}, while {lowest_group} has the "
            f"lowest. The difference is approximately "
            f"{percentage_gap:.2f}%."
        )

        if support:
            finding = f"{finding} {support}"

        evidence = {
            "category": category,
            "metric": metric,
            "highest_group": highest_group,
            "highest_value": highest_value,
            "lowest_group": lowest_group,
            "lowest_value": lowest_value,
            "percentage_gap": percentage_gap,
            "significance": comparison.get(
                "significance",
                confidence,
            ),
        }

        if statistics:
            evidence["statistical_test"] = statistics.get("test")
            evidence["p_value"] = statistics.get("p_value")
            evidence["q_value"] = statistics.get("q_value")
            evidence["effect_size"] = statistics.get("effect_size")
            evidence["effect_metric"] = statistics.get("effect_metric")
            evidence["effect_label"] = statistics.get("effect_label")
            evidence["sample_size"] = statistics.get("n")
            evidence["group_sizes"] = statistics.get("group_sizes")
            evidence["underpowered"] = statistics.get("underpowered")
            evidence["robustness_test"] = statistics.get(
                "robustness_test"
            )
            evidence["robustness_p_value"] = statistics.get(
                "robustness_p_value"
            )

            if statistics.get("note"):
                evidence["caveat"] = statistics["note"]

        insights.append(
            _make_insight(
                insight_type="group_difference",
                priority=priority,
                title=(
                    f"{highest_group} has the highest "
                    f"average {metric}"
                ),
                finding=finding,
                evidence=evidence,
                confidence=confidence,
                metric=str(metric),
                category=str(category),
            )
        )

    return insights


def _correlation_statistics_index(
    analytics: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index the tested correlation pairs by their unordered column pair."""

    index: dict[tuple[str, str], dict[str, Any]] = {}

    tests = analytics.get(
        "correlation_tests",
        [],
    )

    if not isinstance(tests, list):
        return index

    for item in tests:

        if not isinstance(item, dict):
            continue

        statistics = item.get("statistics")

        if not isinstance(statistics, dict):
            continue

        pair = tuple(
            sorted(
                [
                    str(item.get("first_column")),
                    str(item.get("second_column")),
                ]
            )
        )

        index[pair] = statistics

    return index


def _extract_correlation_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract relationships between numeric variables.

    A coefficient above the threshold is necessary but not sufficient: the
    pair must also survive its own significance test after correction for
    the number of pairs examined.
    """

    insights = []

    correlations = analytics.get(
        "correlations",
        {},
    )

    if not isinstance(
        correlations,
        dict,
    ):
        return insights

    statistics_index = _correlation_statistics_index(analytics)

    seen_pairs: set[tuple[str, str]] = set()

    for column, related_columns in correlations.items():

        if not isinstance(
            related_columns,
            dict,
        ):
            continue

        for other_column, value in related_columns.items():

            if column == other_column:
                continue

            if not isinstance(
                value,
                (int, float),
            ):
                continue

            pair = tuple(
                sorted(
                    [
                        str(column),
                        str(other_column),
                    ]
                )
            )

            if pair in seen_pairs:
                continue

            seen_pairs.add(
                pair
            )

            absolute_correlation = abs(
                float(value)
            )

            if absolute_correlation < 0.70:
                continue

            statistics = statistics_index.get(pair, {})

            if statistics:
                confidence = str(
                    statistics.get(
                        "confidence",
                        "medium",
                    )
                )
            elif absolute_correlation >= 0.90:
                confidence = "high"
            else:
                confidence = "medium"

            priority = _priority_from_confidence(confidence)

            direction = (
                "positive"
                if value > 0
                else "negative"
            )

            finding = (
                f"{column} and {other_column} show a "
                f"{direction} correlation of "
                f"{float(value):.2f}."
            )

            support = describe_verdict(statistics)

            if support:
                finding = f"{finding} {support}"

            # Shared variance is the number a reader can act on.
            finding = (
                f"{finding} It accounts for about "
                f"{absolute_correlation ** 2 * 100:.0f}% of the "
                f"variation in either column, and says nothing about "
                f"which one drives the other."
            )

            evidence = {
                "metric_1": str(column),
                "metric_2": str(other_column),
                "correlation": float(value),
                "r_squared": float(absolute_correlation ** 2),
            }

            if statistics:
                evidence["statistical_test"] = statistics.get("test")
                evidence["p_value"] = statistics.get("p_value")
                evidence["q_value"] = statistics.get("q_value")
                evidence["sample_size"] = statistics.get("n")
                evidence["robustness_test"] = statistics.get(
                    "robustness_test"
                )
                evidence["robustness_p_value"] = statistics.get(
                    "robustness_p_value"
                )

                if statistics.get("note"):
                    evidence["caveat"] = statistics["note"]

            insights.append(
                _make_insight(
                    insight_type="correlation",
                    priority=priority,
                    confidence=confidence,
                    title=(
                        f"{column} and {other_column} "
                        f"are strongly correlated"
                    ),
                    finding=finding,
                    evidence=evidence,
                )
            )

    return insights


def _extract_trend_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract trend insights from analytics."""

    insights = []

    trends = analytics.get(
        "trends",
        [],
    )

    if isinstance(
        trends,
        dict,
    ):
        trends = [trends]

    if not isinstance(
        trends,
        list,
    ):
        return insights

    for trend in trends:

        if not isinstance(
            trend,
            dict,
        ):
            continue

        metric = trend.get(
            "metric"
        )

        direction = trend.get(
            "direction"
        )

        if not metric or not direction:
            continue

        priority = (
            "high"
            if direction in {
                "increasing",
                "decreasing",
            }
            else "medium"
        )

        insights.append(
            _make_insight(
                insight_type="trend",
                priority=priority,
                title=(
                    f"{metric} is {direction} over time"
                ),
                finding=(
                    f"{metric} shows a {direction} "
                    f"trend over the observed period."
                ),
                evidence=trend,
                metric=str(metric),
            )
        )

    return insights


def _extract_risk_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract risk insights from analytics."""

    insights = []

    risks = analytics.get(
        "risks",
        [],
    )

    if isinstance(
        risks,
        dict,
    ):
        risks = [risks]

    if not isinstance(
        risks,
        list,
    ):
        return insights

    for risk in risks:

        if not isinstance(
            risk,
            dict,
        ):
            continue

        metric = risk.get(
            "metric"
        )

        risk_level = risk.get(
            "risk_level"
        )

        if not metric or not risk_level:
            continue

        priority = (
            "high"
            if risk_level == "high"
            else "medium"
        )

        insights.append(
            _make_insight(
                insight_type="risk",
                priority=priority,
                title=(
                    f"Potential concentration risk in {metric}"
                ),
                finding=(
                    f"{metric} shows a {risk_level}-level "
                    f"concentration risk."
                ),
                evidence=risk,
                metric=str(metric),
            )
        )

    return insights


def _extract_data_quality_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract data-quality insights."""

    insights = []

    quality = analytics.get(
        "data_quality",
        {},
    )

    if not isinstance(
        quality,
        dict,
    ):
        return insights

    missing_values = quality.get(
        "missing_values",
        {},
    )

    if isinstance(
        missing_values,
        dict,
    ):

        for column, count in missing_values.items():

            if not count:
                continue

            insights.append(
                _make_insight(
                    insight_type="data_quality",
                    priority="medium",
                    title=(
                        f"Missing values detected in {column}"
                    ),
                    finding=(
                        f"{column} contains {count} "
                        f"missing value(s)."
                    ),
                    evidence={
                        "column": column,
                        "missing_count": count,
                    },
                )
            )

    duplicate_rows = quality.get(
        "duplicate_rows",
        0,
    )

    if duplicate_rows:
        insights.append(
            _make_insight(
                insight_type="data_quality",
                priority="medium",
                title="Duplicate rows detected",
                finding=(
                    f"The dataset contains "
                    f"{duplicate_rows} duplicate row(s)."
                ),
                evidence={
                    "duplicate_rows": duplicate_rows,
                },
            )
        )

    return insights

def _extract_categorical_insights(
    analytics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Identify meaningful categorical concentration."""

    insights = []

    summaries = analytics.get(
        "categorical_summary",
        {},
    )

    if not isinstance(
        summaries,
        dict,
    ):
        return insights

    for column, summary in summaries.items():

        if not isinstance(
            summary,
            dict,
        ):
            continue

        top_value = summary.get(
            "top_value"
        )

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
                confidence="high",
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

def generate_insights(
    analytics: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate meaningful, dataset-agnostic insights
    from universal analytics results.
    """

    if not isinstance(
        analytics,
        dict,
    ):
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

    # IMPORTANT:
    # Restore category-concentration insights.
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
            item.get(
                "priority",
                "medium",
            ),
            1,
        )
    )

    return {
        "insights": insights,
        "insight_count": len(
            insights
        ),
    }