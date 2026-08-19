import pandas as pd

from src.analytics import (
    generate_business_summary,
)

from src.insights import (
    generate_business_insights,
)


def build_ai_context(df: pd.DataFrame) -> dict:
    """Build structured business context for an AI model."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    business_summary = generate_business_summary(df)
    business_insights = generate_business_insights(df)

    return {
        "business_summary": business_summary,
        "business_insights": business_insights,
    }