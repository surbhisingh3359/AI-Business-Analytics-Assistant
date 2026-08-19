import pandas as pd


def _count_missing_values(df: pd.DataFrame) -> int:
    """Count all missing cells in a DataFrame."""

    return int(df.isna().sum().sum())


def _get_basic_metrics(df: pd.DataFrame) -> dict:
    """Calculate basic data-quality metrics."""

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": _count_missing_values(df),
    }


def generate_quality_report(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> dict:
    """Compare a DataFrame before and after preprocessing."""

    if not isinstance(before_df, pd.DataFrame):
        raise TypeError("before_df must be a pandas DataFrame.")

    if not isinstance(after_df, pd.DataFrame):
        raise TypeError("after_df must be a pandas DataFrame.")

    before = _get_basic_metrics(before_df)
    after = _get_basic_metrics(after_df)

    return {
        "before": before,
        "after": after,
        "changes": {
            "rows_removed": before["rows"] - after["rows"],
            "columns_removed": before["columns"] - after["columns"],
            "missing_values_resolved": (
                before["missing_values"] - after["missing_values"]
            ),
        },
    }