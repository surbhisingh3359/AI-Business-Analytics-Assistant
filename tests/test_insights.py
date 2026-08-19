import pandas as pd
import pytest

from src.insights import (analyze_top_category_contribution,
                          analyze_top_product_concentration,
                          generate_product_concentration_insight,
                          generate_business_insights,
)


def test_analyze_top_category_contribution():
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

    result = analyze_top_category_contribution(df)

    assert result["category"] == "Electronics"
    assert result["revenue"] == 118500.0
    assert result["contribution_percentage"] == 87.13


def test_analyze_top_category_contribution_zero_revenue():
    df = pd.DataFrame(
        {
            "order_id": [1001],
            "product": ["Test"],
            "category": ["Test"],
            "quantity": [0],
            "price": [100],
        }
    )

    result = analyze_top_category_contribution(df)

    assert result["contribution_percentage"] == 0.0


def test_analyze_top_category_contribution_invalid_input():
    with pytest.raises(TypeError):
        analyze_top_category_contribution("not a dataframe")

def test_analyze_top_product_concentration():
    df = pd.DataFrame(
        {
            "product": [
                "Laptop",
                "Mouse",
                "Keyboard",
                "Chair",
                "Desk",
            ],
            "quantity": [2, 5, 3, 2, 1],
            "price": [55000, 800, 1500, 4500, 8500],
        }
    )

    result = analyze_top_product_concentration(df)

    assert result["product"] == "Laptop"
    assert result["revenue"] == 110000.0
    assert result["concentration_percentage"] == 80.88
    assert result["risk_level"] == "high"


def test_analyze_top_product_concentration_zero_revenue():
    df = pd.DataFrame(
        {
            "product": ["Test"],
            "quantity": [0],
            "price": [100],
        }
    )

    result = analyze_top_product_concentration(df)

    assert result["concentration_percentage"] == 0.0
    assert result["risk_level"] == "low"


def test_analyze_top_product_concentration_invalid_input():
    with pytest.raises(TypeError):
        analyze_top_product_concentration("not a dataframe")


def test_product_concentration_moderate():
    df = pd.DataFrame(
        {
            "product": ["A", "B"],
            "quantity": [4, 6],
            "price": [100, 100],
        }
    )

    result = analyze_top_product_concentration(df)

    assert result["concentration_percentage"] == 60.0
    assert result["risk_level"] == "moderate"


def test_product_concentration_low():
    df = pd.DataFrame(
        {
            "product": ["A", "B", "C", "D"],
            "quantity": [1, 1, 1, 1],
            "price": [100, 100, 100, 100],
        }
    )

    result = analyze_top_product_concentration(df)

    assert result["concentration_percentage"] == 25.0
    assert result["risk_level"] == "low"


def test_generate_product_concentration_insight():
    df = pd.DataFrame(
        {
            "product": ["Laptop", "Mouse"],
            "quantity": [2, 5],
            "price": [55000, 800],
        }
    )

    result = generate_product_concentration_insight(df)

    assert "High Product Concentration" in result
    assert "Laptop" in result
    assert "110,000.00" in result
    assert "96.49%" in result

def test_generate_business_insights():
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

    result = generate_business_insights(df)

    assert "top_category" in result
    assert "top_product" in result
    assert "product_concentration_insight" in result

    assert result["top_category"]["category"] == "Electronics"
    assert result["top_product"]["product"] == "Laptop"
    assert "High Product Concentration" in (
        result["product_concentration_insight"]
    )


def test_generate_business_insights_invalid_input():
    with pytest.raises(TypeError):
        generate_business_insights("not a dataframe")