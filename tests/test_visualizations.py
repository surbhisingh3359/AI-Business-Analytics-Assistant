import pandas as pd
import pytest
from src.visualizations import (
    revenue_by_category,
    revenue_by_product,
    quantity_by_product,
)


@pytest.fixture
def sales_data():
    return pd.DataFrame(
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


def test_revenue_by_category(sales_data):
    result = revenue_by_category(sales_data)

    assert result.iloc[0]["category"] == "Electronics"
    assert result.iloc[0]["revenue"] == 118500.0


def test_revenue_by_product(sales_data):
    result = revenue_by_product(sales_data)

    assert result.iloc[0]["product"] == "Laptop"
    assert result.iloc[0]["revenue"] == 110000.0


def test_quantity_by_product(sales_data):
    result = quantity_by_product(sales_data)

    assert result.iloc[0]["product"] == "Mouse"
    assert result.iloc[0]["quantity"] == 5