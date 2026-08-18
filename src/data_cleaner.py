import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform safe, deterministic cleaning operations."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned_df = df.copy()

    # Remove completely empty rows
    cleaned_df = cleaned_df.dropna(how="all")

    # Remove completely empty columns
    cleaned_df = cleaned_df.dropna(axis=1, how="all")

    # Remove exact duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()

    # Reset the index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df