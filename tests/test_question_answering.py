
import pandas as pd
import pytest

from src.question_answering import interpret_question


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "Product": ["Laptop", "Chair", "Laptop", "Chair"],
        "Revenue": [1000, 500, 1500, 700],
        "Quantity": [2, 1, 3, 2],
        "Region": ["East", "West", "East", "West"],
    })


def test_interpret_group_comparison(sample_dataframe):
    result = interpret_question(
        "Which Product generated the highest Revenue?",
        sample_dataframe,
    )

    assert result["analysis_type"] == "group_comparison"
    assert result["metric"] == "Revenue"
    assert result["dimension"] == "Product"
    assert result["status"] == "planned"


def test_interpret_descriptive_analysis(sample_dataframe):
    result = interpret_question(
        "Show me a summary of the dataset",
        sample_dataframe,
    )

    assert result["analysis_type"] == "descriptive_analysis"
    assert result["status"] == "planned"


def test_reject_question_without_numeric_metric(sample_dataframe):
    with pytest.raises(ValueError, match="numeric metric"):
        interpret_question(
            "Compare Product across Region",
            sample_dataframe,
        )


def test_reject_question_without_categorical_dimension(sample_dataframe):
    with pytest.raises(ValueError, match="categorical dimension"):
        interpret_question(
            "Compare Revenue and Quantity",
            sample_dataframe,
        )


def test_reject_multiple_numeric_columns(sample_dataframe):
    with pytest.raises(ValueError, match="Multiple numeric columns"):
        interpret_question(
            "Compare Revenue by Quantity and Product",
            sample_dataframe,
        )


def test_reject_empty_question(sample_dataframe):
    with pytest.raises(ValueError, match="question cannot be empty"):
        interpret_question("", sample_dataframe)


def test_reject_invalid_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        interpret_question(
            "Show summary",
            None,
        )


def test_reject_empty_dataframe():
    with pytest.raises(ValueError, match="empty DataFrame"):
        interpret_question(
            "Show summary",
            pd.DataFrame(),
        )


def test_reject_unsupported_question(sample_dataframe):
    with pytest.raises(ValueError, match="Unable to identify"):
        interpret_question(
            "Tell me something interesting",
            sample_dataframe,
        )
