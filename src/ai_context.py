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
    Build the legacy business_summary and business_insights
    structures required by the existing prompt and validation
    pipeline.

    This is intentionally used as a compatibility layer.

    It activates only when the dataset contains recognizable
    business columns. The main analytics pipeline remains
    dataset-agnostic.
    """

    summary: dict[str, Any] = {
        "total_orders": int(len(df)),
    }

    insights: dict[str, Any] = {
        "top_category": None,
        "top_product": None,
    }

    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    # ---------------------------------------------------------
    # Quantity
    # ---------------------------------------------------------

    quantity_column = None

    for name in (
        "quantity",
        "qty",
        "units",
        "unit_quantity",
    ):
        if name in columns:
            quantity_column = columns[name]
            break

    if quantity_column is not None:

        quantity_values = pd.to_numeric(
            df[quantity_column],
            errors="coerce",
        )

        summary["total_quantity"] = int(
            quantity_values.fillna(0).sum()
        )

    else:
        summary["total_quantity"] = 0

    # ---------------------------------------------------------
    # Monetary value
    # ---------------------------------------------------------

    price_column = None

    for name in (
        "price",
        "revenue",
        "sales",
        "amount",
        "value",
    ):
        if name in columns:
            price_column = columns[name]
            break

    if price_column is not None:

        price_values = pd.to_numeric(
            df[price_column],
            errors="coerce",
        ).fillna(0)

        # If quantity exists, calculate line-level revenue.
        if quantity_column is not None:

            quantity_values = pd.to_numeric(
                df[quantity_column],
                errors="coerce",
            ).fillna(0)

            revenue = (
                quantity_values * price_values
            )

        else:
            revenue = price_values

        summary["total_revenue"] = float(
            revenue.sum()
        )

    # ---------------------------------------------------------
    # Top category
    # ---------------------------------------------------------

    category_column = None

    for name in (
        "category",
        "product_category",
        "department",
        "segment",
        "group",
    ):
        if name in columns:
            category_column = columns[name]
            break

    if category_column is not None:

        category_data = df[
            [category_column]
        ].copy()

        if quantity_column is not None:
            category_data["_quantity"] = pd.to_numeric(
                df[quantity_column],
                errors="coerce",
            ).fillna(0)

        if price_column is not None:
            category_data["_price"] = pd.to_numeric(
                df[price_column],
                errors="coerce",
            ).fillna(0)

        if (
            quantity_column is not None
            and price_column is not None
        ):
            category_data["_value"] = (
                category_data["_quantity"]
                * category_data["_price"]
            )

            category_totals = (
                category_data
                .groupby(category_column)["_value"]
                .sum()
                .sort_values(ascending=False)
            )

        elif quantity_column is not None:

            category_totals = (
                category_data
                .groupby(category_column)["_quantity"]
                .sum()
                .sort_values(ascending=False)
            )

        elif price_column is not None:

            category_totals = (
                category_data
                .groupby(category_column)["_price"]
                .sum()
                .sort_values(ascending=False)
            )

        else:

            category_totals = (
                df[category_column]
                .value_counts()
            )

        if not category_totals.empty:

            top_category = category_totals.index[0]

            insights["top_category"] = {
                "category": str(top_category),
                "value": float(
                    category_totals.iloc[0]
                ),
            }

    # ---------------------------------------------------------
    # Top product
    # ---------------------------------------------------------

    product_column = None

    for name in (
        "product",
        "product_name",
        "item",
        "item_name",
    ):
        if name in columns:
            product_column = columns[name]
            break

    if product_column is not None:

        product_data = df[
            [product_column]
        ].copy()

        if quantity_column is not None:
            product_data["_quantity"] = pd.to_numeric(
                df[quantity_column],
                errors="coerce",
            ).fillna(0)

        if price_column is not None:
            product_data["_price"] = pd.to_numeric(
                df[price_column],
                errors="coerce",
            ).fillna(0)

        # Prefer revenue/value when both quantity and price
        # are available.
        if (
            quantity_column is not None
            and price_column is not None
        ):

            product_data["_value"] = (
                product_data["_quantity"]
                * product_data["_price"]
            )

            product_totals = (
                product_data
                .groupby(product_column)["_value"]
                .sum()
                .sort_values(ascending=False)
            )

        elif quantity_column is not None:

            product_totals = (
                product_data
                .groupby(product_column)["_quantity"]
                .sum()
                .sort_values(ascending=False)
            )

        elif price_column is not None:

            product_totals = (
                product_data
                .groupby(product_column)["_price"]
                .sum()
                .sort_values(ascending=False)
            )

        else:

            product_totals = (
                df[product_column]
                .value_counts()
            )

        if not product_totals.empty:

            top_product = product_totals.index[0]

            insights["top_product"] = {
                "product": str(top_product),
                "value": float(
                    product_totals.iloc[0]
                ),
            }

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