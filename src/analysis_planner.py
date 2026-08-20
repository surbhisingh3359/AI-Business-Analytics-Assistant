from typing import Any

import pandas as pd


def build_analysis_plan(
    df: pd.DataFrame,
    semantic_roles: dict[str, str],
) -> dict[str, Any]:
    """
    Build a dataset-agnostic analysis plan.

    The planner examines detected semantic roles and determines
    which analyses are appropriate for the uploaded dataset.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if not isinstance(semantic_roles, dict):
        raise TypeError("semantic_roles must be a dictionary.")

    if df.empty:
        raise ValueError("Cannot build an analysis plan for an empty dataset.")

    analyses: list[str] = []
    columns_by_role: dict[str, list[str]] = {}

    # ---------------------------------------------------------
    # Organize columns by semantic role
    # ---------------------------------------------------------

    for column, role in semantic_roles.items():
        columns_by_role.setdefault(role, []).append(column)

    numeric_columns = columns_by_role.get("numeric", [])
    monetary_columns = columns_by_role.get("monetary", [])
    quantity_columns = columns_by_role.get("quantity", [])
    percentage_columns = columns_by_role.get("percentage", [])
    categorical_columns = columns_by_role.get("categorical", [])
    datetime_columns = columns_by_role.get("datetime", [])
    age_columns = columns_by_role.get("age", [])
    location_columns = columns_by_role.get("location", [])

    # ---------------------------------------------------------
    # Universal dataset analysis
    # ---------------------------------------------------------

    analyses.extend(
        [
            "dataset_overview",
            "data_quality",
        ]
    )

    # ---------------------------------------------------------
    # Numeric analysis
    # ---------------------------------------------------------

    if numeric_columns or monetary_columns or quantity_columns:
        analyses.append("numeric_summary")

    # ---------------------------------------------------------
    # Distribution analysis
    # ---------------------------------------------------------

    if numeric_columns or monetary_columns or quantity_columns:
        analyses.append("distribution_analysis")

    # ---------------------------------------------------------
    # Categorical analysis
    # ---------------------------------------------------------

    if categorical_columns:
        analyses.append("categorical_analysis")

    # ---------------------------------------------------------
    # Date / time analysis
    # ---------------------------------------------------------

    if datetime_columns:
        analyses.append("time_series_analysis")

    # ---------------------------------------------------------
    # Monetary analysis
    # ---------------------------------------------------------

    if monetary_columns:
        analyses.append("monetary_analysis")

    # ---------------------------------------------------------
    # Quantity analysis
    # ---------------------------------------------------------

    if quantity_columns:
        analyses.append("quantity_analysis")

    # ---------------------------------------------------------
    # Percentage analysis
    # ---------------------------------------------------------

    if percentage_columns:
        analyses.append("percentage_analysis")

    # ---------------------------------------------------------
    # Age analysis
    # ---------------------------------------------------------

    if age_columns:
        analyses.append("age_analysis")

    # ---------------------------------------------------------
    # Location analysis
    # ---------------------------------------------------------

    if location_columns:
        analyses.append("location_analysis")

    # ---------------------------------------------------------
    # Relationship analysis
    # ---------------------------------------------------------

    quantitative_roles = (
        numeric_columns
        + monetary_columns
        + quantity_columns
        + percentage_columns
        + age_columns
    )

    if len(quantitative_roles) >= 2:
        analyses.append("correlation_analysis")

    # ---------------------------------------------------------
    # Categorical vs numeric analysis
    # ---------------------------------------------------------

    if categorical_columns and quantitative_roles:
        analyses.append("group_comparison_analysis")

    # ---------------------------------------------------------
    # Time + numeric analysis
    # ---------------------------------------------------------

    if datetime_columns and quantitative_roles:
        analyses.append("trend_analysis")

    # ---------------------------------------------------------
    # Dataset characteristics
    # ---------------------------------------------------------

    characteristics = {
        "has_numeric_data": bool(
            numeric_columns
            or monetary_columns
            or quantity_columns
            or percentage_columns
            or age_columns
        ),
        "has_categorical_data": bool(categorical_columns),
        "has_datetime_data": bool(datetime_columns),
        "has_monetary_data": bool(monetary_columns),
        "has_quantity_data": bool(quantity_columns),
        "has_percentage_data": bool(percentage_columns),
        "has_age_data": bool(age_columns),
        "has_location_data": bool(location_columns),
    }

    return {
        "analyses": analyses,
        "columns_by_role": columns_by_role,
        "characteristics": characteristics,
    }