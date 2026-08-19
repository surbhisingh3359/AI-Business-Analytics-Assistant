import pandas as pd


def calculate_total_revenue(
    df: pd.DataFrame,
    quantity_column: str = "quantity",
    price_column: str = "price",
) -> float:
    """Calculate total revenue from quantity multiplied by price."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = {quantity_column, price_column}

    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    revenue = df[quantity_column] * df[price_column]

    return float(revenue.sum())

def calculate_total_orders(
    df: pd.DataFrame,
    order_id_column: str = "order_id",
) -> int:
    """Calculate the total number of unique orders."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if order_id_column not in df.columns:
        raise ValueError(
            f"Missing required column: {order_id_column}"
        )

    return int(df[order_id_column].nunique())

def calculate_total_quantity(
    df: pd.DataFrame,
    quantity_column: str = "quantity",
) -> int:
    """Calculate the total quantity sold."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if quantity_column not in df.columns:
        raise ValueError(
            f"Missing required column: {quantity_column}"
        )

    return int(df[quantity_column].sum())


def calculate_average_order_value(
    df: pd.DataFrame,
    quantity_column: str = "quantity",
    price_column: str = "price",
    order_id_column: str = "order_id",
) -> float:
    """Calculate average revenue generated per unique order."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = {
        quantity_column,
        price_column,
        order_id_column,
    }

    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    total_revenue = calculate_total_revenue(
        df,
        quantity_column=quantity_column,
        price_column=price_column,
    )

    total_orders = calculate_total_orders(
        df,
        order_id_column=order_id_column,
    )

    if total_orders == 0:
        return 0.0

    return total_revenue / total_orders


def calculate_top_product(
    df: pd.DataFrame,
    product_column: str = "product",
    quantity_column: str = "quantity",
    price_column: str = "price",
) -> dict:
    """Return the product with the highest total revenue."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = {
        product_column,
        quantity_column,
        price_column,
    }

    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        return {
            "product": None,
            "revenue": 0.0,
        }

    revenue = df[quantity_column] * df[price_column]

    product_revenue = (
        df.assign(_revenue=revenue)
        .groupby(product_column)["_revenue"]
        .sum()
    )

    top_product = product_revenue.idxmax()

    return {
        "product": top_product,
        "revenue": float(product_revenue[top_product]),
    }

def calculate_top_category(
    df: pd.DataFrame,
    category_column: str = "category",
    quantity_column: str = "quantity",
    price_column: str = "price",
) -> dict:
    """Return the category with the highest total revenue."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = {
        category_column,
        quantity_column,
        price_column,
    }

    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        return {
            "category": None,
            "revenue": 0.0,
        }

    revenue = df[quantity_column] * df[price_column]

    category_revenue = (
        df.assign(_revenue=revenue)
        .groupby(category_column)["_revenue"]
        .sum()
    )

    top_category = category_revenue.idxmax()

    return {
        "category": top_category,
        "revenue": float(category_revenue[top_category]),
    }

def generate_business_summary(df: pd.DataFrame) -> dict:
    """Generate a complete business summary from sales data."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    return {
        "total_revenue": calculate_total_revenue(df),
        "total_orders": calculate_total_orders(df),
        "total_quantity": calculate_total_quantity(df),
        "average_order_value": calculate_average_order_value(df),
        "top_product": calculate_top_product(df),
        "top_category": calculate_top_category(df),
    }