from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def _normalized_name(column: Any) -> str:
    """Return a normalized lowercase column name."""
    return (
        str(column)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _is_date_like(series: pd.Series) -> bool:
    """
    Determine whether a column contains mostly date-like values.

    Uses a conservative threshold to avoid classifying arbitrary
    categorical text as datetime.
    """

    sample = series.dropna().astype(str).head(100)

    if sample.empty:
        return False

    parsed = pd.to_datetime(
        sample,
        errors="coerce",
        format="mixed",
    )

    return parsed.notna().mean() >= 0.80


def _has_keyword(column_name: str, keywords: list[str]) -> bool:
    """Check whether a column name contains one of the supplied keywords."""

    parts = set(
        column_name
        .replace("-", "_")
        .split("_")
    )

    return any(
        keyword in parts
        or keyword in column_name
        for keyword in keywords
    )


def detect_identifier_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Detect columns that are likely identifiers.

    Date-like columns are explicitly excluded.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    identifier_columns: list[str] = []

    identifier_keywords = [
        "id",
        "code",
        "key",
        "uuid",
        "identifier",
        "number",
        "no",
    ]

    for column in df.columns:

        name = _normalized_name(column)
        series = df[column]

        # -----------------------------------------------------
        # Never classify datetime columns as identifiers
        # -----------------------------------------------------

        if is_datetime64_any_dtype(series):
            continue

        if not is_numeric_dtype(series):
            if _is_date_like(series):
                continue

        # -----------------------------------------------------
        # Name-based detection
        # -----------------------------------------------------

        if _has_keyword(name, identifier_keywords):

            # A name such as "order_id" or "employee_id"
            # is a strong identifier signal.
            identifier_columns.append(column)
            continue

        # -----------------------------------------------------
        # Value-based detection
        # -----------------------------------------------------

        non_null = series.dropna()

        if non_null.empty:
            continue

        unique_ratio = (
            non_null.nunique(dropna=True)
            / len(non_null)
        )

        # High-cardinality text is often an identifier.
        if (
            unique_ratio >= 0.95
            and not is_numeric_dtype(series)
        ):
            identifier_columns.append(column)

    return identifier_columns


def detect_semantic_roles(
    df: pd.DataFrame,
) -> dict[str, str]:
    """
    Detect semantic roles for dataset columns.

    Supported roles:

    identifier
    datetime
    monetary
    quantity
    percentage
    age
    location
    boolean
    categorical
    numeric
    unknown
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        return {}

    semantic_roles: dict[str, str] = {}

    identifier_columns = set(
        detect_identifier_columns(df)
    )

    # ---------------------------------------------------------
    # Semantic keyword groups
    # ---------------------------------------------------------

    monetary_keywords = [
        "price",
        "revenue",
        "sales",
        "income",
        "salary",
        "wage",
        "cost",
        "profit",
        "amount",
        "expense",
        "budget",
        "fee",
        "payment",
        "spend",
        "value",
        "turnover",
        "earning",
        "earnings",
    ]

    quantity_keywords = [
        "quantity",
        "qty",
        "units",
        "count",
        "volume",
        "number_of",
        "num_",
        "total_units",
    ]

    percentage_keywords = [
        "percentage",
        "percent",
        "pct",
        "rate",
        "ratio",
        "margin",
        "accuracy",
        "attendance",
        "conversion",
        "completion",
    ]

    age_keywords = [
        "age",
    ]

    location_keywords = [
        "city",
        "state",
        "country",
        "location",
        "region",
        "address",
        "zip",
        "zipcode",
        "postal",
        "postcode",
    ]

    datetime_keywords = [
        "date",
        "time",
        "timestamp",
        "datetime",
        "created_at",
        "updated_at",
        "modified_at",
        "joining_date",
        "purchase_date",
        "order_date",
        "transaction_date",
        "registration_date",
        "start_date",
        "end_date",
    ]

    # ---------------------------------------------------------
    # Analyze columns
    # ---------------------------------------------------------

    for column in df.columns:

        name = _normalized_name(column)
        series = df[column]

        # -----------------------------------------------------
        # 1. Explicit datetime dtype
        # -----------------------------------------------------

        if is_datetime64_any_dtype(series):
            semantic_roles[column] = "datetime"
            continue

        # -----------------------------------------------------
        # 2. Date-like strings
        # -----------------------------------------------------

        if not is_numeric_dtype(series):

            if _is_date_like(series):
                semantic_roles[column] = "datetime"
                continue

        # -----------------------------------------------------
        # 3. Date/time column name
        # -----------------------------------------------------

        if _has_keyword(name, datetime_keywords):

            # Only classify as datetime if the values are
            # actually parseable or the column is empty.
            if series.dropna().empty or _is_date_like(series):
                semantic_roles[column] = "datetime"
                continue

        # -----------------------------------------------------
        # 4. Boolean
        # -----------------------------------------------------

        if is_bool_dtype(series):
            semantic_roles[column] = "boolean"
            continue

        # -----------------------------------------------------
        # 5. Location
        # -----------------------------------------------------

        if _has_keyword(name, location_keywords):
            semantic_roles[column] = "location"
            continue

        # -----------------------------------------------------
        # 6. Identifier
        # -----------------------------------------------------

        if column in identifier_columns:
            semantic_roles[column] = "identifier"
            continue

        # -----------------------------------------------------
        # The following semantic roles require numeric data.
        # -----------------------------------------------------

        if is_numeric_dtype(series):

            # -------------------------------------------------
            # 7. Monetary
            # -------------------------------------------------

            if _has_keyword(name, monetary_keywords):
                semantic_roles[column] = "monetary"
                continue

            # -------------------------------------------------
            # 8. Quantity
            # -------------------------------------------------

            if _has_keyword(name, quantity_keywords):
                semantic_roles[column] = "quantity"
                continue

            # -------------------------------------------------
            # 9. Percentage
            # -------------------------------------------------

            if _has_keyword(name, percentage_keywords):
                semantic_roles[column] = "percentage"
                continue

            # -------------------------------------------------
            # 10. Age
            # -------------------------------------------------

            if _has_keyword(name, age_keywords):
                semantic_roles[column] = "age"
                continue

            # -------------------------------------------------
            # 11. Generic numeric
            # -------------------------------------------------

            semantic_roles[column] = "numeric"
            continue

        # -----------------------------------------------------
        # 12. String / categorical
        # -----------------------------------------------------

        if pd.api.types.is_string_dtype(series):
            semantic_roles[column] = "categorical"
            continue

        # -----------------------------------------------------
        # 13. Unknown
        # -----------------------------------------------------

        semantic_roles[column] = "unknown"

    return semantic_roles