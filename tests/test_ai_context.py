import pandas as pd
import pytest

from src.ai_context import build_ai_context


def test_build_ai_context():
    df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1003, 1004, 1005],
            "product": [
                "Laptop",
                "Mouse",
                "Keyboard",
                "Chair",
                "Desk",
            ],
            "category": [
                "Electronics",
                "Electronics",
                "Electronics",
                "Furniture",
                "Furniture",
            ],
            "quantity": [2, 5, 3, 2, 1],
            "price": [55000, 800, 1500, 4500, 8500],
        }
    )

    result = build_ai_context(df)

    assert "business_summary" in result
    assert "business_insights" in result

    assert result["business_summary"]["total_orders"] == 5
    assert result["business_summary"]["total_quantity"] == 13

    assert (
        result["business_insights"]["top_category"]["category"]
        == "Electronics"
    )

    assert (
        result["business_insights"]["top_product"]["product"]
        == "Laptop"
    )


def test_build_ai_context_invalid_input():
    with pytest.raises(TypeError):
        build_ai_context("not a dataframe")


def test_build_ai_context_includes_business_questions():

    df = pd.DataFrame(
        {
            "department": [
                "IT",
                "IT",
                "HR",
                "HR",
                "Finance",
                "Finance",
            ],
            "salary": [
                90000,
                95000,
                50000,
                52000,
                65000,
                68000,
            ],
        }
    )

    result = build_ai_context(df)

    assert "business_questions" in result
    assert isinstance(
        result["business_questions"],
        list,
    )