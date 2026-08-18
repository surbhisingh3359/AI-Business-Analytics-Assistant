from src.data_cleaner import clean_data, fill_missing_values
from src.data_loader import load_data
from src.data_profiler import profile_data
from src.schema_analyzer import analyze_schema


def preprocess_data(file_path: str) -> dict:
    """Load, profile, analyze, and clean a dataset."""

    df = load_data(file_path)

    profile = profile_data(df)

    schema = analyze_schema(df)

    cleaned_df = clean_data(df)

    cleaned_df = fill_missing_values(cleaned_df)

    return {
        "data": cleaned_df,
        "profile": profile,
        "schema": schema,
    }