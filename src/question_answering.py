
import re
from typing import Any

import pandas as pd

from src.analysis_execution import build_execution_plan


# =========================================================
# QUESTION NORMALIZATION
# =========================================================

def _normalize_text(text: str) -> str:
    """Normalize text for consistent column matching."""

    return " ".join(
        re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).split()
    )


# =========================================================
# COLUMN IDENTIFICATION
# =========================================================

def _find_mentioned_columns(
    question: str,
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Identify DataFrame columns explicitly mentioned
    in the user's question.
    """

    normalized_question = f" {_normalize_text(question)} "

    matches = []

    for column in dataframe.columns:
        normalized_column = _normalize_text(str(column))

        if not normalized_column:
            continue

        column_phrase = f" {normalized_column} "

        if column_phrase in normalized_question:
            matches.append(str(column))

    return matches


def _get_column_by_type(
    columns: list[str],
    dataframe: pd.DataFrame,
    numeric: bool,
) -> list[str]:
    """Separate explicitly mentioned columns by data type."""

    matched = []

    for column in columns:
        is_numeric = pd.api.types.is_numeric_dtype(
            dataframe[column]
        )

        is_datetime = pd.api.types.is_datetime64_any_dtype(
            dataframe[column]
        )

        if numeric and is_numeric and not is_datetime:
            matched.append(column)

        elif not numeric and not is_numeric and not is_datetime:
            matched.append(column)

    return matched


# =========================================================
# QUESTION INTENT
# =========================================================

def _identify_analysis_type(question: str) -> str:
    """Identify the supported analytical operation."""

    normalized = _normalize_text(question)

    descriptive_terms = {
        "summary",
        "summarize",
        "describe",
        "overview",
        "dataset overview",
    }

    if any(
        term in normalized
        for term in descriptive_terms
    ):
        return "descriptive_analysis"

    comparison_terms = {
        "highest",
        "lowest",
        "most",
        "least",
        "compare",
        "comparison",
        "across",
        "by",
        "per",
        "breakdown",
    }

    if any(
        term in normalized.split()
        for term in comparison_terms
    ):
        return "group_comparison"

    raise ValueError(
        "Unable to identify the requested analysis. "
        "Currently supported: descriptive summaries "
        "and group comparisons."
    )


# =========================================================
# QUESTION INTERPRETATION
# =========================================================

def interpret_question(
    question: str,
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Convert a supported natural-language question into
    a validated execution plan.

    Supported operations:
        - descriptive_analysis
        - group_comparison

    Group comparison currently uses SUM aggregation
    through the existing execution engine.
    """

    # -----------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------

    if not isinstance(question, str):
        raise TypeError("question must be a string.")

    if not question.strip():
        raise ValueError("question cannot be empty.")

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Cannot analyze an empty DataFrame.")

    # -----------------------------------------------------
    # Identify intent
    # -----------------------------------------------------

    analysis_type = _identify_analysis_type(question)

    # -----------------------------------------------------
    # Descriptive analysis
    # -----------------------------------------------------

    if analysis_type == "descriptive_analysis":

        decision = {
            "decision": question,
            "recommended_analysis": analysis_type,
            "metrics": [],
            "dimensions": [],
            "source_type": "user_question",
        }

        return build_execution_plan(decision)

    # -----------------------------------------------------
    # Identify explicitly mentioned columns
    # -----------------------------------------------------

    mentioned_columns = _find_mentioned_columns(
        question,
        dataframe,
    )

    numeric_columns = _get_column_by_type(
        mentioned_columns,
        dataframe,
        numeric=True,
    )

    categorical_columns = _get_column_by_type(
        mentioned_columns,
        dataframe,
        numeric=False,
    )

   
    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------

    # First, check whether required column types exist.

    if len(numeric_columns) == 0:
        raise ValueError(
            "No numeric metric column was identified. "
            "Mention a numeric column from your dataset."
        )

    if len(categorical_columns) == 0:
        raise ValueError(
            "No categorical dimension was identified. "
            "Mention a group column from your dataset."
        )

    # -----------------------------------------------------
    # Validate column ambiguity
    # -----------------------------------------------------

    if len(numeric_columns) > 1:
        raise ValueError(
            "Multiple numeric columns were identified: "
            f"{numeric_columns}. Please specify one metric."
        )

    if len(categorical_columns) > 1:
        raise ValueError(
            "Multiple categorical columns were identified: "
            f"{categorical_columns}. Please specify one dimension."
        )

    # -----------------------------------------------------
    # Assign validated columns
    # -----------------------------------------------------

    metric = numeric_columns[0]
    dimension = categorical_columns[0]

    # -----------------------------------------------------
    # Build compatible execution plan
    # -----------------------------------------------------

    decision = {
        "decision": question,
        "recommended_analysis": analysis_type,
        "metrics": [metric],
        "dimensions": [dimension],
        "source_type": "user_question",
    }

    return build_execution_plan(decision)
