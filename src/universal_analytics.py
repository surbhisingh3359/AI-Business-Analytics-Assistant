from typing import Any

import pandas as pd


def _numeric_summary(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, float]]:
    """Generate descriptive statistics for numeric columns."""

    result: dict[str, dict[str, float]] = {}

    for column in columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            continue

        result[column] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
        }

    return result


def _categorical_summary(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Generate frequency summaries for categorical columns."""

    result: dict[str, dict[str, Any]] = {}

    for column in columns:

        series = df[column].dropna()

        if series.empty:
            continue

        counts = series.astype(str).value_counts()

        result[column] = {
            "unique_values": int(series.nunique()),
            "top_value": str(counts.index[0]),
            "top_value_count": int(counts.iloc[0]),
            "top_value_percentage": float(
                counts.iloc[0] / len(series) * 100
            ),
        }

    return result


def _datetime_summary(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Generate time-range summaries."""

    result: dict[str, dict[str, Any]] = {}

    for column in columns:

        dates = pd.to_datetime(
            df[column],
            errors="coerce",
        ).dropna()

        if dates.empty:
            continue

        result[column] = {
            "min_date": str(dates.min()),
            "max_date": str(dates.max()),
            "unique_dates": int(dates.nunique()),
        }

    return result


def _correlation_analysis(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, float]]:
    """Calculate correlations between quantitative variables."""

    if len(columns) < 2:
        return {}

    numeric_df = df[columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation = numeric_df.corr()

    result: dict[str, dict[str, float]] = {}

    for column in correlation.columns:
        result[column] = {}

        for other_column in correlation.columns:

            value = correlation.loc[column, other_column]

            if pd.notna(value):
                result[column][other_column] = float(value)

    return result


def _group_comparison(
    df: pd.DataFrame,
    categorical_columns: list[str],
    quantitative_columns: list[str],
    max_categories: int = 10,
    max_comparisons: int = 5,
) -> list[dict[str, Any]]:
    """
    Generate concise, high-value group comparisons.

    Instead of returning every categorical × quantitative
    combination, this function identifies the strongest
    meaningful differences.

    Each result contains:
        - category
        - metric
        - highest group
        - lowest group
        - highest value
        - lowest value
        - percentage gap
        - significance
    """

    if not categorical_columns or not quantitative_columns:
        return []

    candidates: list[dict[str, Any]] = []

    for category in categorical_columns:

        if category not in df.columns:
            continue

        category_series = df[category]

        unique_count = category_series.dropna().nunique()

        if unique_count < 2 or unique_count > max_categories:
            continue

        for metric in quantitative_columns:

            if metric not in df.columns:
                continue

            values = pd.to_numeric(
                df[metric],
                errors="coerce",
            )

            comparison_df = pd.DataFrame(
                {
                    "category": category_series,
                    "value": values,
                }
            ).dropna()

            if comparison_df.empty:
                continue

            grouped = (
                comparison_df
                .groupby("category")["value"]
                .agg(["mean", "median", "count"])
                .sort_values("mean", ascending=False)
            )

            if len(grouped) < 2:
                continue

            highest_group = grouped["mean"].idxmax()
            lowest_group = grouped["mean"].idxmin()

            highest_value = float(
                grouped.loc[highest_group, "mean"]
            )

            lowest_value = float(
                grouped.loc[lowest_group, "mean"]
            )

            # -------------------------------------------------
            # Calculate percentage difference
            # -------------------------------------------------

            denominator = abs(lowest_value)

            if denominator == 0:

                if highest_value == 0:
                    percentage_gap = 0.0
                else:
                    percentage_gap = 100.0

            else:

                percentage_gap = (
                    abs(highest_value - lowest_value)
                    / denominator
                    * 100
                )

            # Ignore insignificant differences.
            if percentage_gap < 5:
                continue

            # -------------------------------------------------
            # Determine significance
            # -------------------------------------------------

            if percentage_gap >= 50:
                significance = "high"

            elif percentage_gap >= 20:
                significance = "medium"

            else:
                significance = "low"

            candidates.append(
                {
                    "category": str(category),
                    "metric": str(metric),
                    "group_count": int(len(grouped)),
                    "highest_group": str(highest_group),
                    "highest_value": highest_value,
                    "lowest_group": str(lowest_group),
                    "lowest_value": lowest_value,
                    "percentage_gap": float(
                        percentage_gap
                    ),
                    "significance": significance,
                }
            )

    # ---------------------------------------------------------
    # Rank comparisons by importance
    # ---------------------------------------------------------

    significance_weight = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    candidates.sort(
        key=lambda item: (
            significance_weight.get(
                item["significance"],
                0,
            ),
            item["percentage_gap"],
        ),
        reverse=True,
    )

    # ---------------------------------------------------------
    # Avoid repeating the same category/metric relationship
    # ---------------------------------------------------------

    selected: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for candidate in candidates:

        pair = (
            candidate["category"],
            candidate["metric"],
        )

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)
        selected.append(candidate)

        if len(selected) >= max_comparisons:
            break

    return selected

def analyze_dataset(
    df: pd.DataFrame,
    semantic_roles: dict[str, str],
    analysis_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a dataset-agnostic analysis plan.

    This function does not assume that the dataset is a
    sales, employee, student, healthcare, or marketing dataset.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    if not isinstance(semantic_roles, dict):
        raise TypeError(
            "semantic_roles must be a dictionary."
        )

    if not isinstance(analysis_plan, dict):
        raise TypeError(
            "analysis_plan must be a dictionary."
        )

    columns_by_role = analysis_plan.get(
        "columns_by_role",
        {},
    )

    numeric_columns = columns_by_role.get(
        "numeric",
        [],
    )

    monetary_columns = columns_by_role.get(
        "monetary",
        [],
    )

    quantity_columns = columns_by_role.get(
        "quantity",
        [],
    )

    percentage_columns = columns_by_role.get(
        "percentage",
        [],
    )

    age_columns = columns_by_role.get(
        "age",
        [],
    )

    categorical_columns = columns_by_role.get(
        "categorical",
        [],
    )

    datetime_columns = columns_by_role.get(
        "datetime",
        [],
    )

    quantitative_columns = (
        numeric_columns
        + monetary_columns
        + quantity_columns
        + percentage_columns
        + age_columns
    )

    results: dict[str, Any] = {
        "dataset_overview": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [
                str(column)
                for column in df.columns
            ],
        },
        "data_quality": {
            "missing_values": {
                str(column): int(
                    df[column].isna().sum()
                )
                for column in df.columns
                if df[column].isna().sum() > 0
            },
            "duplicate_rows": int(
                df.duplicated().sum()
            ),
        },
        "numeric_summary": {},
        "categorical_summary": {},
        "datetime_summary": {},
        "correlations": {},
        "group_comparisons": {},
    }

    analyses = analysis_plan.get(
        "analyses",
        [],
    )

    if "numeric_summary" in analyses:
        results["numeric_summary"] = _numeric_summary(
            df,
            quantitative_columns,
        )

    if "categorical_analysis" in analyses:
        results["categorical_summary"] = _categorical_summary(
            df,
            categorical_columns,
        )

    if "time_series_analysis" in analyses:
        results["datetime_summary"] = _datetime_summary(
            df,
            datetime_columns,
        )

    if "correlation_analysis" in analyses:
        results["correlations"] = _correlation_analysis(
            df,
            quantitative_columns,
        )

    if "group_comparison_analysis" in analyses:
        results["group_comparisons"] = _group_comparison(
            df,
            categorical_columns,
            quantitative_columns,
        )

    results["role_summary"] = {
        role: list(columns)
        for role, columns in columns_by_role.items()
    }

    return results