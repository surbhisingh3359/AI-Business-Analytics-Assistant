import pandas as pd


def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Return total revenue grouped by category."""

    required_columns = {"category", "quantity", "price"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = (
        df.assign(revenue=df["quantity"] * df["price"])
        .groupby("category", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )

    return result


def revenue_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Return total revenue grouped by product."""

    required_columns = {"product", "quantity", "price"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = (
        df.assign(revenue=df["quantity"] * df["price"])
        .groupby("product", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )

    return result


def quantity_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """Return total quantity sold by product."""

    required_columns = {"product", "quantity"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = (
        df.groupby("product", as_index=False)["quantity"]
        .sum()
        .sort_values("quantity", ascending=False)
    )

    return result