import pandas as pd

from src.analytics import (
    calculate_top_category,
    calculate_top_product,
    calculate_total_revenue,
)

def analyze_top_category_contribution(
    df: pd.DataFrame,
) -> dict:
    """Analyze the revenue contribution of the top category."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    total_revenue = calculate_total_revenue(df)
    top_category = calculate_top_category(df)

    if total_revenue == 0:
        return {
            "category": top_category["category"],
            "revenue": top_category["revenue"],
            "contribution_percentage": 0.0,
        }

    contribution_percentage = (
        top_category["revenue"] / total_revenue
    ) * 100

    return {
        "category": top_category["category"],
        "revenue": top_category["revenue"],
        "contribution_percentage": round(
            contribution_percentage, 2
        ),
    }

def analyze_top_product_concentration(
    df: pd.DataFrame,
) -> dict:
    """Analyze revenue concentration of the top product."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    total_revenue = calculate_total_revenue(df)
    top_product = calculate_top_product(df)

    if total_revenue == 0:
        return {
            "product": top_product["product"],
            "revenue": top_product["revenue"],
            "concentration_percentage": 0.0,
            "risk_level": "low",
        }

    concentration_percentage = (
        top_product["revenue"] / total_revenue
    ) * 100

    if concentration_percentage > 60:
        risk_level = "high"
    elif concentration_percentage >= 30:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "product": top_product["product"],
        "revenue": top_product["revenue"],
        "concentration_percentage": round(
            concentration_percentage, 2
        ),
        "risk_level": risk_level,
    }

def generate_product_concentration_insight(
    df: pd.DataFrame,
) -> str:
    """Generate a human-readable product concentration insight."""

    result = analyze_top_product_concentration(df)

    product = result["product"]
    revenue = result["revenue"]
    percentage = result["concentration_percentage"]
    risk_level = result["risk_level"]

    if risk_level == "high":
        return (
            f"High Product Concentration: {product} generates "
            f"₹{revenue:,.2f}, representing {percentage:.2f}% "
            f"of total revenue. The business may be highly "
            f"dependent on this product."
        )

    if risk_level == "moderate":
        return (
            f"Moderate Product Concentration: {product} generates "
            f"₹{revenue:,.2f}, representing {percentage:.2f}% "
            f"of total revenue."
        )

    return (
        f"Low Product Concentration: {product} generates "
        f"₹{revenue:,.2f}, representing {percentage:.2f}% "
        f"of total revenue. Revenue appears relatively diversified."
    )

def generate_business_insights(df: pd.DataFrame) -> dict:
    """Generate a complete set of business insights."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    return {
        "top_category": analyze_top_category_contribution(df),
        "top_product": analyze_top_product_concentration(df),
        "product_concentration_insight": (
            generate_product_concentration_insight(df)
        ),
    }