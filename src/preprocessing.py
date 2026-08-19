from src.data_cleaner import clean_data, fill_missing_values
from src.data_loader import load_data
from src.data_profiler import profile_data
from src.data_quality import generate_quality_report
from src.schema_analyzer import analyze_schema


def preprocess_data(file_path: str) -> dict:
    """Load, profile, analyze, and clean a dataset."""

    original_df = load_data(file_path)

    profile = profile_data(original_df)

    schema = analyze_schema(original_df)

    cleaned_df = clean_data(original_df)

    cleaned_df = fill_missing_values(cleaned_df)

    quality_report = generate_quality_report(
        original_df,
        cleaned_df,
    )

    return {
        "data": cleaned_df,
        "profile": profile,
        "schema": schema,
        "quality_report": quality_report,
    }