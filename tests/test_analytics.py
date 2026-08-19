import pandas as pd
import pytest

from src.analytics import (
    calculate_total_orders,
    calculate_total_quantity,
    calculate_total_revenue,
    calculate_average_order_value,
    calculate_top_product,
    calculate_top_category,
    generate_business_summary,
)

def test_calculate_total_revenue():
    df = pd.DataFrame(
        {
            "quantity": [2, 5, 3],
            "price": [55000, 800, 1500],
        }
    )

    result = calculate_total_revenue(df)

    assert result == 118500.0


def test_calculate_total_revenue_custom_columns():
    df = pd.DataFrame(
        {
            "units_sold": [2, 5],
            "unit_price": [100, 50],
        }
    )

    result = calculate_total_revenue(
        df,
        quantity_column="units_sold",
        price_column="unit_price",
    )

    assert result == 450.0


def test_calculate_total_revenue_invalid_input():
    with pytest.raises(TypeError):
        calculate_total_revenue("not a dataframe")


def test_calculate_total_revenue_missing_columns():
    df = pd.DataFrame(
        {
            "quantity": [2, 5],
        }
    )

    with pytest.raises(ValueError):
        calculate_total_revenue(df)

def test_calculate_total_orders():
    df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1002, 1003],
        }
    )

    result = calculate_total_orders(df)

    assert result == 3


def test_calculate_total_orders_custom_column():
    df = pd.DataFrame(
        {
            "transaction_id": ["A", "B", "B", "C"],
        }
    )

    result = calculate_total_orders(
        df,
        order_id_column="transaction_id",
    )

    assert result == 3


def test_calculate_total_orders_invalid_input():
    with pytest.raises(TypeError):
        calculate_total_orders("not a dataframe")


def test_calculate_total_orders_missing_column():
    df = pd.DataFrame(
        {
            "product": ["Laptop", "Mouse"],
        }
    )

    with pytest.raises(ValueError):
        calculate_total_orders(df)


def test_calculate_total_quantity():
    df = pd.DataFrame(
        {
            "quantity": [2, 5, 3, 2, 1],
        }
    )

    result = calculate_total_quantity(df)

    assert result == 13


def test_calculate_total_quantity_custom_column():
    df = pd.DataFrame(
        {
            "units": [10, 20, 5],
        }
    )

    result = calculate_total_quantity(
        df,
        quantity_column="units",
    )

    assert result == 35


def test_calculate_total_quantity_invalid_input():
    with pytest.raises(TypeError):
        calculate_total_quantity("not a dataframe")


def test_calculate_total_quantity_missing_column():
    df = pd.DataFrame(
        {
            "product": ["Laptop", "Mouse"],
        }
    )

    with pytest.raises(ValueError):
        calculate_total_quantity(df)


def test_calculate_average_order_value():
    df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1003],
            "quantity": [2, 5, 3],
            "price": [100, 50, 20],
        }
    )

    result = calculate_average_order_value(df)

    assert result == 170.0


def test_calculate_average_order_value_with_duplicate_orders():
    df = pd.DataFrame(
        {
            "order_id": [1001, 1001, 1002],
            "quantity": [2, 1, 2],
            "price": [100, 50, 50],
        }
    )

    result = calculate_average_order_value(df)

    assert result == 175.0


def test_calculate_average_order_value_zero_orders():
    df = pd.DataFrame(
        {
            "order_id": [],
            "quantity": [],
            "price": [],
        }
    )

    result = calculate_average_order_value(df)

    assert result == 0.0


def test_calculate_average_order_value_invalid_input():
    with pytest.raises(TypeError):
        calculate_average_order_value("not a dataframe")

def test_calculate_top_product():
    df = pd.DataFrame(
        {
            "product": ["Laptop", "Mouse", "Keyboard"],
            "quantity": [2, 5, 3],
            "price": [55000, 800, 1500],
        }
    )

    result = calculate_top_product(df)

    assert result["product"] == "Laptop"
    assert result["revenue"] == 110000.0


def test_calculate_top_product_custom_columns():
    df = pd.DataFrame(
        {
            "item": ["A", "B", "C"],
            "units": [2, 10, 3],
            "unit_price": [100, 20, 50],
        }
    )

    result = calculate_top_product(
        df,
        product_column="item",
        quantity_column="units",
        price_column="unit_price",
    )

    assert result["product"] == "A"
    assert result["revenue"] == 200.0


def test_calculate_top_product_empty_dataframe():
    df = pd.DataFrame(
        columns=["product", "quantity", "price"]
    )

    result = calculate_top_product(df)

    assert result["product"] is None
    assert result["revenue"] == 0.0


def test_calculate_top_product_invalid_input():
    with pytest.raises(TypeError):
        calculate_top_product("not a dataframe")


def test_calculate_top_category():
    df = pd.DataFrame(
        {
            "category": [
                "Electronics",
                "Electronics",
                "Furniture",
            ],
            "quantity": [2, 5, 2],
            "price": [55000, 800, 4500],
        }
    )

    result = calculate_top_category(df)

    assert result["category"] == "Electronics"
    assert result["revenue"] == 114000.0


def test_calculate_top_category_custom_columns():
    df = pd.DataFrame(
        {
            "type": ["A", "B", "A"],
            "units": [2, 10, 3],
            "unit_price": [100, 20, 50],
        }
    )

    result = calculate_top_category(
        df,
        category_column="type",
        quantity_column="units",
        price_column="unit_price",
    )

    assert result["category"] == "A"
    assert result["revenue"] == 350.0


def test_calculate_top_category_empty_dataframe():
    df = pd.DataFrame(
        columns=["category", "quantity", "price"]
    )

    result = calculate_top_category(df)

    assert result["category"] is None
    assert result["revenue"] == 0.0


def test_calculate_top_category_invalid_input():
    with pytest.raises(TypeError):
        calculate_top_category("not a dataframe")

def test_generate_business_summary():
    df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1003],
            "product": ["Laptop", "Mouse", "Keyboard"],
            "category": ["Electronics", "Electronics", "Electronics"],
            "quantity": [2, 5, 3],
            "price": [55000, 800, 1500],
        }
    )

    result = generate_business_summary(df)

    assert result["total_revenue"] == 118500.0
    assert result["total_orders"] == 3
    assert result["total_quantity"] == 10
    assert result["average_order_value"] == 39500.0

    assert result["top_product"]["product"] == "Laptop"
    assert result["top_product"]["revenue"] == 110000.0

    assert result["top_category"]["category"] == "Electronics"
    assert result["top_category"]["revenue"] == 118500.0


def test_generate_business_summary_invalid_input():
    with pytest.raises(TypeError):
        generate_business_summary("not a dataframe")