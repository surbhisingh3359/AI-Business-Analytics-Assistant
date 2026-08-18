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

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using basic type-aware strategies."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned_df = df.copy()

    numeric_columns = cleaned_df.select_dtypes(
        include="number"
    ).columns

    categorical_columns = cleaned_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in numeric_columns:
        if cleaned_df[column].isnull().any():
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].median()
            )

    for column in categorical_columns:
        if cleaned_df[column].isnull().any():
            cleaned_df[column] = cleaned_df[column].fillna("Unknown")

    return cleaned_df