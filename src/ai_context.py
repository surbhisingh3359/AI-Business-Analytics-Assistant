from typing import Any

import pandas as pd

from src.analysis_planner import build_analysis_plan
from src.dataset_classifier import detect_semantic_roles
from src.universal_analytics import analyze_dataset
from src.insight_engine import generate_insights
from src.insight_ranker import rank_insights
from src.business_questions import generate_business_questions


def _build_business_compatibility_context(
    df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build backward-compatible business_summary and business_insights.

    This layer is intentionally adaptive.

    It supports common schemas such as:

    Sales:
        order_id, quantity, price

    Online Retail:
        InvoiceNo, StockCode, Description,
        Quantity, UnitPrice, InvoiceDate, CustomerID

    HR:
        employee_id, department, salary

    Generic datasets:
        numeric and categorical columns

    The universal analytics pipeline remains dataset-agnostic.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        return (
            {
                "total_revenue": 0.0,
                "total_orders": 0,
                "total_quantity": 0,
                "average_order_value": 0.0,
            },
            {
                "top_category": None,
                "top_product": None,
            },
        )

    # ---------------------------------------------------------
    # Normalize column names for detection
    # ---------------------------------------------------------

    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    def find_column(*names: str):
        """
        Return the first matching real DataFrame column.
        """
        for name in names:
            normalized = name.strip().lower()

            if normalized in columns:
                return columns[normalized]

        return None

    # ---------------------------------------------------------
    # Detect common semantic columns
    # ---------------------------------------------------------

    order_column = find_column(
        "order_id",
        "orderid",
        "invoice_no",
        "invoiceno",
        "invoice",
        "transaction_id",
        "transactionid",
        "receipt_id",
    )

    quantity_column = find_column(
        "quantity",
        "qty",
        "units",
        "unit_quantity",
        "units_sold",
        "quantity_sold",
    )

    price_column = find_column(
        "price",
        "unitprice",
        "unit_price",
        "unit cost",
        "unit_cost",
        "revenue",
        "sales",
        "amount",
        "value",
        "total_price",
        "total_sales",
    )

    product_column = find_column(
        "product",
        "product_name",
        "productname",
        "item",
        "item_name",
        "description",
        "stockcode",
        "stock_code",
        "sku",
        "product_id",
        "productid",
    )

    category_column = find_column(
        "category",
        "product_category",
        "department",
        "segment",
        "group",
        "class",
        "type",
    )

    # ---------------------------------------------------------
    # Detect revenue
    # ---------------------------------------------------------

    revenue = None

    if price_column is not None:

        price_values = pd.to_numeric(
            df[price_column],
            errors="coerce",
        ).fillna(0)

        if quantity_column is not None:

            quantity_values = pd.to_numeric(
                df[quantity_column],
                errors="coerce",
            ).fillna(0)

            revenue = (
                quantity_values
                * price_values
            )

        else:

            revenue = price_values

    # ---------------------------------------------------------
    # Total quantity
    # ---------------------------------------------------------

    if quantity_column is not None:

        quantity_values = pd.to_numeric(
            df[quantity_column],
            errors="coerce",
        ).fillna(0)

        total_quantity = float(
            quantity_values.sum()
        )

    else:

        total_quantity = 0.0

    # ---------------------------------------------------------
    # Total orders
    # ---------------------------------------------------------

    if order_column is not None:

        order_values = df[order_column].dropna()

        # Count unique orders/invoices rather than rows.
        total_orders = int(
            order_values.nunique()
        )

    else:

        # Backward-compatible fallback.
        total_orders = int(len(df))

    # ---------------------------------------------------------
    # Total revenue
    # ---------------------------------------------------------

    if revenue is not None:

        total_revenue = float(
            revenue.sum()
        )

    else:

        total_revenue = 0.0

    # ---------------------------------------------------------
    # Average order value
    # ---------------------------------------------------------

    if total_orders > 0:

        average_order_value = (
            total_revenue
            / total_orders
        )

    else:

        average_order_value = 0.0

    # ---------------------------------------------------------
    # Business summary
    # ---------------------------------------------------------

    summary: dict[str, Any] = {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_quantity": int(total_quantity),
        "average_order_value": float(
            average_order_value
        ),
    }

    # ---------------------------------------------------------
    # Initialize insights
    # ---------------------------------------------------------

    insights: dict[str, Any] = {
        "top_category": None,
        "top_product": None,
    }

    # =========================================================
    # TOP CATEGORY
    # =========================================================

    if category_column is not None:

        category_data = pd.DataFrame(
            {
                "category": df[category_column]
            }
        )

        # Revenue-based category analysis
        if revenue is not None:

            category_data["value"] = (
                revenue.values
            )

        # Quantity-based category analysis
        elif quantity_column is not None:

            category_data["value"] = (
                quantity_values.values
            )

        # Frequency-based category analysis
        else:

            category_data["value"] = 1.0

        category_data = category_data.dropna(
            subset=["category"]
        )

        if not category_data.empty:

            category_totals = (
                category_data
                .groupby("category")["value"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            if not category_totals.empty:

                top_category = (
                    category_totals.index[0]
                )

                top_category_value = float(
                    category_totals.iloc[0]
                )

                if total_revenue > 0 and revenue is not None:

                    contribution_percentage = (
                        top_category_value
                        / total_revenue
                        * 100
                    )

                elif (
                    total_quantity > 0
                    and quantity_column is not None
                ):

                    contribution_percentage = (
                        top_category_value
                        / total_quantity
                        * 100
                    )

                else:

                    contribution_percentage = (
                        top_category_value
                        / len(df)
                        * 100
                    )

                insights["top_category"] = {
                    "category": str(
                        top_category
                    ),
                    "value": top_category_value,
                    "revenue": (
                        top_category_value
                        if revenue is not None
                        else 0.0
                    ),
                    "contribution_percentage": float(
                        contribution_percentage
                    ),
                }

    # =========================================================
    # TOP PRODUCT
    # =========================================================

    if product_column is not None:

        product_data = pd.DataFrame(
            {
                "product": df[product_column]
            }
        )

        # Revenue-based product analysis
        if revenue is not None:

            product_data["value"] = (
                revenue.values
            )

        # Quantity-based product analysis
        elif quantity_column is not None:

            product_data["value"] = (
                quantity_values.values
            )

        # Frequency-based product analysis
        else:

            product_data["value"] = 1.0

        product_data = product_data.dropna(
            subset=["product"]
        )

        if not product_data.empty:

            product_totals = (
                product_data
                .groupby("product")["value"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            if not product_totals.empty:

                top_product = (
                    product_totals.index[0]
                )

                top_product_value = float(
                    product_totals.iloc[0]
                )

                if total_revenue > 0 and revenue is not None:

                    concentration_percentage = (
                        top_product_value
                        / total_revenue
                        * 100
                    )

                    risk_level = (
                        "high"
                        if concentration_percentage >= 50
                        else "medium"
                        if concentration_percentage >= 25
                        else "low"
                    )

                elif (
                    total_quantity > 0
                    and quantity_column is not None
                ):

                    concentration_percentage = (
                        top_product_value
                        / total_quantity
                        * 100
                    )

                    risk_level = (
                        "high"
                        if concentration_percentage >= 50
                        else "medium"
                        if concentration_percentage >= 25
                        else "low"
                    )

                else:

                    concentration_percentage = (
                        top_product_value
                        / len(df)
                        * 100
                    )

                    risk_level = "unknown"

                insights["top_product"] = {
                    "product": str(
                        top_product
                    ),
                    "value": top_product_value,
                    "revenue": (
                        top_product_value
                        if revenue is not None
                        else 0.0
                    ),
                    "concentration_percentage": float(
                        concentration_percentage
                    ),
                    "risk_level": risk_level,
                }

                if revenue is not None:

                    insights[
                        "product_concentration_insight"
                    ] = (
                        f"Top product '{top_product}' "
                        f"generates "
                        f"₹{top_product_value:,.2f}, "
                        f"representing "
                        f"{concentration_percentage:.2f}% "
                        f"of total revenue."
                    )

    return summary, insights

def build_ai_context(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a complete, dataset-aware AI context.

    Pipeline:

        Dataset
            ↓
        Semantic classification
            ↓
        Analysis planning
            ↓
        Universal analytics
            ↓
        Insight generation
            ↓
        Insight ranking
            ↓
        Business question generation
            ↓
        AI context

    The universal analytics layer remains dataset-agnostic.

    Legacy business_summary and business_insights fields are
    retained for compatibility with the existing prompt and
    response-validation pipeline.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot build AI context from an empty dataset."
        )

    # ---------------------------------------------------------
    # 1. Detect semantic roles
    # ---------------------------------------------------------

    semantic_roles = detect_semantic_roles(df)

    # ---------------------------------------------------------
    # 2. Build adaptive analysis plan
    # ---------------------------------------------------------

    analysis_plan = build_analysis_plan(
        df,
        semantic_roles,
    )

    # ---------------------------------------------------------
    # 3. Execute universal analytics
    # ---------------------------------------------------------

    analytics = analyze_dataset(
        df,
        semantic_roles,
        analysis_plan,
    )

    # ---------------------------------------------------------
    # 4. Generate candidate insights
    # ---------------------------------------------------------

    insight_result = generate_insights(
        analytics,
    )

    candidate_insights = insight_result.get(
        "insights",
        [],
    )

    # ---------------------------------------------------------
    # 5. Rank and deduplicate insights
    # ---------------------------------------------------------

    ranked_insights = rank_insights(
        candidate_insights,
        max_insights=5,
    )

    # ---------------------------------------------------------
    # 6. Generate business questions
    # ---------------------------------------------------------

    business_questions = generate_business_questions(
        ranked_insights,
        max_questions=5,
    )

    # ---------------------------------------------------------
    # 7. Build backward-compatible business context
    # ---------------------------------------------------------

    business_summary, business_insights = (
        _build_business_compatibility_context(df)
    )

    # ---------------------------------------------------------
    # 8. Dataset metadata
    # ---------------------------------------------------------

    dataset_info = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [
            str(column)
            for column in df.columns
        ],
    }

    # ---------------------------------------------------------
    # 9. Return complete AI context
    # ---------------------------------------------------------

    return {
        # Dataset metadata
        "dataset": dataset_info,

        # Universal analysis
        "semantic_roles": semantic_roles,
        "analysis_plan": analysis_plan,
        "analytics": analytics,

        # Ranked analytical insights
        "insights": ranked_insights,
        "insight_count": len(ranked_insights),

        # New business-question layer
        "business_questions": business_questions,
        "business_question_count": len(
            business_questions
        ),

        # Backward-compatible prompt fields
        "business_summary": business_summary,
        "business_insights": business_insights,
    }