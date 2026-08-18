import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


IDENTIFIER_KEYWORDS = (
    "id",
    "_id",
)


def _is_identifier(column_name: str) -> bool:
    """Determine whether a column name looks like an identifier."""

    name = column_name.lower().strip()

    return name == "id" or name.endswith("_id")


def _classify_column(column_name: str, series: pd.Series) -> str:
    """Classify a column according to its likely analytical role."""

    if is_bool_dtype(series):
        return "boolean"

    if is_datetime64_any_dtype(series):
        return "date"

    if _is_identifier(column_name):
        return "identifier"

    if is_numeric_dtype(series):
        return "numeric"

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    ):
        return "categorical"

    return "unknown"


def analyze_schema(df: pd.DataFrame) -> dict:
    """Analyze DataFrame columns and determine their likely roles."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    schema = {}

    for column in df.columns:
        schema[column] = {
            "dtype": str(df[column].dtype),
            "role": _classify_column(column, df[column]),
        }

    return schema