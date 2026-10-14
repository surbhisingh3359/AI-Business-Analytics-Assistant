from typing import Any


def _safe_dict(value: Any) -> dict[str, Any]:
    """Return value as a dictionary or an empty dictionary."""

    if isinstance(value, dict):
        return value

    return {}


def _format_number(value: Any) -> str:
    """Safely format numeric values."""

    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _build_legacy_prompt(
    context: dict[str, Any],
) -> str:
    """
    Build the original business-analysis prompt.

    This keeps backward compatibility with older callers and tests
    that provide business_summary and business_insights.
    """

    summary = context["business_summary"]
    insights = context["business_insights"]

    if not isinstance(summary, dict):
        raise TypeError(
            "business_summary must be a dictionary."
        )

    if not isinstance(insights, dict):
        raise TypeError(
            "business_insights must be a dictionary."
        )

    total_revenue = float(
        summary.get("total_revenue", 0)
    )

    total_orders = int(
        summary.get("total_orders", 0)
    )

    total_quantity = int(
        summary.get("total_quantity", 0)
    )

    average_order_value = summary.get(
        "average_order_value"
    )

    if average_order_value is None:
        if total_orders > 0:
            average_order_value = (
                total_revenue / total_orders
            )
        else:
            average_order_value = 0.0

    average_order_value = float(
        average_order_value
    )

    top_category = insights.get(
        "top_category"
    )

    if not isinstance(top_category, dict):
        top_category = {
            "category": "N/A",
            "revenue": 0.0,
            "contribution_percentage": 0.0,
        }

    top_product = insights.get(
        "top_product"
    )

    if not isinstance(top_product, dict):
        top_product = {
            "product": "N/A",
            "revenue": 0.0,
            "concentration_percentage": 0.0,
            "risk_level": "unknown",
        }

    category_name = str(
        top_category.get(
            "category",
            "N/A",
        )
    )

    category_revenue = float(
        top_category.get(
            "revenue",
            0,
        )
    )

    category_contribution = float(
        top_category.get(
            "contribution_percentage",
            0,
        )
    )

    product_name = str(
        top_product.get(
            "product",
            "N/A",
        )
    )

    product_revenue = float(
        top_product.get(
            "revenue",
            0,
        )
    )

    product_concentration = float(
        top_product.get(
            "concentration_percentage",
            0,
        )
    )

    product_risk = str(
        top_product.get(
            "risk_level",
            "unknown",
        )
    )

    concentration_insight = str(
        insights.get(
            "product_concentration_insight",
            "No product concentration insight was provided.",
        )
    )

    return f"""
You are an expert business analyst.

Analyze the business data provided below and produce a concise,
fact-based business analysis for a decision-maker.

IMPORTANT GROUNDING RULES
-------------------------
1. Use ONLY information explicitly provided below.
2. Do NOT invent facts, statistics, customer behavior, market conditions,
   competitors, supply-chain conditions, profitability, or other information
   that is not present in the data.
3. Do NOT claim that customer demand is increasing or decreasing.
4. Do NOT claim anything about customer preferences or market penetration.
5. Do NOT claim anything about competition or market saturation.
6. Do NOT claim anything about suppliers, inventory, stock availability,
   quality problems, or geographic performance.
7. Risks must be directly supported by the provided metrics.
8. Recommendations must be clearly presented as suggestions.
9. Include important numerical metrics.
10. Do not change, estimate, or reinterpret provided numbers.

BUSINESS METRICS
----------------
Total Revenue: ₹{total_revenue:,.2f}
Total Orders: {total_orders}
Total Quantity Sold: {total_quantity}
Average Order Value: ₹{average_order_value:,.2f}

TOP CATEGORY
------------
Category: {category_name}
Revenue: ₹{category_revenue:,.2f}
Revenue Contribution: {category_contribution:.2f}%

TOP PRODUCT
-----------
Product: {product_name}
Revenue: ₹{product_revenue:,.2f}
Revenue Concentration: {product_concentration:.2f}%
Risk Level: {product_risk}

PRODUCT CONCENTRATION INSIGHT
-----------------------------
{concentration_insight}

REQUIRED OUTPUT
---------------
Provide exactly these four sections:

### Executive Summary

Briefly summarize the overall business performance using the
provided metrics.

### Most Important Business Finding

Identify the most important finding supported by the data.

### Potential Business Risk

Explain a risk directly supported by the provided metrics.

Do not introduce external assumptions.

### Actionable Recommendations

Provide exactly two actionable recommendations based ONLY
on the observed data.

Recommendations must directly address the provided metrics
or patterns.

Clearly present recommendations as suggestions.

FINAL CHECK
-----------
Before answering, verify that:

- Total Revenue is mentioned.
- Total Orders is mentioned.
- Total Quantity Sold is mentioned.
- Average Order Value is mentioned.
- The top product is mentioned.
- The top category is mentioned.
- No unsupported external claims are introduced.
""".strip()


def _build_universal_prompt(
    context: dict[str, Any],
) -> str:
    """
    Build a universal dataset-aware business analysis prompt.

    This version does not assume that the dataset represents sales.
    It can work with employee, finance, education, healthcare,
    customer, operational, or other structured datasets.
    """

    dataset = _safe_dict(
        context.get("dataset")
    )

    semantic_roles = _safe_dict(
        context.get("semantic_roles")
    )

    analysis_plan = _safe_dict(
        context.get("analysis_plan")
    )

    analytics = _safe_dict(
        context.get("analytics")
    )

    insights = context.get(
        "insights",
        []
    )

    business_questions = context.get(
        "business_questions",
        []
    )

    rows = dataset.get(
        "rows",
        0,
    )

    columns = dataset.get(
        "columns",
        0,
    )

    column_names = dataset.get(
        "column_names",
        [],
    )

    return f"""
You are an expert AI business analyst.

Your task is to analyze the provided dataset and produce a
grounded, decision-useful business analysis.

This is a UNIVERSAL analytics system.

IMPORTANT:
The dataset is NOT necessarily a sales dataset.

It may represent:
- sales
- customers
- employees
- finance
- operations
- education
- healthcare
- products
- transactions
- marketing
- logistics
- or another structured business domain.

You MUST infer the domain from the supplied schema, semantic roles,
analytics, insights, and business questions.

Do not assume sales terminology unless the supplied data supports it.

============================================================
DATASET
============================================================

Rows: {rows}
Columns: {columns}

Column Names:
{column_names}

============================================================
SEMANTIC ROLES
============================================================

{semantic_roles}

============================================================
ANALYSIS PLAN
============================================================

{analysis_plan}

============================================================
ANALYTICS
============================================================

{analytics}

============================================================
DETECTED INSIGHTS
============================================================

{insights}

============================================================
BUSINESS QUESTIONS
============================================================

{business_questions}

============================================================
GROUNDING RULES
============================================================

1. Use ONLY information provided in this context.

2. Do NOT invent statistics.

3. Do NOT invent customer behavior.

4. Do NOT invent market conditions.

5. Do NOT invent competitors.

6. Do NOT invent profitability unless profitability metrics
   are explicitly provided.

7. Do NOT assume that a numeric column represents revenue.

8. Do NOT assume that rows represent orders.

9. Do NOT assume that a column represents customers.

10. Do NOT claim causation from correlation.

11. Clearly distinguish observations from recommendations.

12. When evidence is insufficient, explicitly state that
    the available data is insufficient.

13. Numerical values must remain consistent with the supplied
    analytics.

14. Business recommendations must be based on observed patterns.

15. Avoid generic recommendations that are unrelated to the
    detected data.

============================================================
REQUIRED ANALYSIS
============================================================

### Executive Summary

Summarize the most important characteristics and findings
from the dataset.

### Dataset Understanding

Briefly explain what the dataset appears to represent based
only on its schema and detected semantic roles.

### Most Important Business Finding

Identify the strongest finding supported by the analytics.

### Key Insights

Present the most important detected insights.

### Business Questions

Address the supplied business questions when sufficient
evidence exists.

### Potential Business Risks

Identify risks only when they are directly supported by
the observed data.

### Actionable Recommendations

Provide practical recommendations based only on the
observed evidence.

Clearly label recommendations as suggestions rather than facts.

============================================================
FINAL VALIDATION
============================================================

Before producing the final answer verify that:

- The dataset domain was inferred rather than assumed.
- Important supplied metrics are preserved.
- Detected insights are not contradicted.
- No unsupported external claims are introduced.
- No causal relationship is claimed without evidence.
- Recommendations are grounded in the supplied data.
""".strip()


def build_business_prompt(
    context: dict[str, Any],
) -> str:
    """
    Build a business-analysis prompt.

    Supports both:

    1. Legacy context:
       business_summary + business_insights

    2. Universal context:
       dataset
       semantic_roles
       analysis_plan
       analytics
       insights
       business_questions

    This backward-compatible design allows the existing test
    suite and the new universal analytics pipeline to coexist.
    """

    if not isinstance(
        context,
        dict,
    ):
        raise TypeError(
            "context must be a dictionary."
        )

    # ---------------------------------------------------------
    # Legacy context
    # ---------------------------------------------------------

    if (
        "business_summary" in context
        and "business_insights" in context
    ):
        return _build_legacy_prompt(
            context
        )

    # ---------------------------------------------------------
    # Universal context
    # ---------------------------------------------------------

    required_fields = [
        "dataset",
        "semantic_roles",
        "analysis_plan",
        "analytics",
        "insights",
        "business_questions",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in context
    ]

    if missing_fields:
        raise ValueError(
            "Missing required context fields: "
            + ", ".join(missing_fields)
        )

    return _build_universal_prompt(
        context
    )