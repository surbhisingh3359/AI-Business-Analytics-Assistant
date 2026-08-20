from typing import Any


def build_business_prompt(context: dict[str, Any]) -> str:
    """Build a grounded AI prompt from structured business context."""

    if not isinstance(context, dict):
        raise TypeError("context must be a dictionary.")

    if "business_summary" not in context:
        raise ValueError("Missing business_summary in context.")

    if "business_insights" not in context:
        raise ValueError("Missing business_insights in context.")

    summary = context["business_summary"]
    insights = context["business_insights"]

    if not isinstance(summary, dict):
        raise TypeError("business_summary must be a dictionary.")

    if not isinstance(insights, dict):
        raise TypeError("business_insights must be a dictionary.")

    # ---------------------------------------------------------
    # Safely retrieve business metrics
    # ---------------------------------------------------------

    total_revenue = float(
        summary.get("total_revenue", 0)
    )

    total_orders = int(
        summary.get("total_orders", 0)
    )

    total_quantity = int(
        summary.get("total_quantity", 0)
    )

    # Average order value may not exist in older contexts.
    # Derive it safely from available metrics.
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

    # ---------------------------------------------------------
    # Safely retrieve top category/product information
    # ---------------------------------------------------------

    top_category = insights.get(
        "top_category"
    )

    top_product = insights.get(
        "top_product"
    )

    if not isinstance(top_category, dict):
        top_category = {
            "category": "N/A",
            "revenue": 0.0,
            "contribution_percentage": 0.0,
        }

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

    product_concentration_insight = str(
        insights.get(
            "product_concentration_insight",
            "No product concentration insight was provided.",
        )
    )

    # ---------------------------------------------------------
    # Build grounded prompt
    # ---------------------------------------------------------

    prompt = f"""
You are an expert business analyst.

Analyze the business data provided below and produce a concise,
fact-based business analysis for a decision-maker.

IMPORTANT GROUNDING RULES
-------------------------
1. Use ONLY information explicitly provided in the business data below.
2. Do NOT invent facts, statistics, customer behavior, market conditions,
   competitors, supply-chain conditions, profitability, or other information
   that is not present in the provided data.
3. Do NOT claim that customer demand is increasing or decreasing.
4. Do NOT claim anything about customer preferences or market penetration.
5. Do NOT claim anything about competition or market saturation.
6. Do NOT claim anything about suppliers, inventory, stock availability,
   quality problems, or geographic performance.
7. You MAY identify a business risk when that risk is directly supported
   by the provided metrics.
8. When making recommendations, clearly present them as suggestions
   rather than facts.
9. Include important numerical metrics in the analysis.
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
{product_concentration_insight}

REQUIRED OUTPUT
---------------
Provide exactly these four sections:

### Executive Summary
Briefly summarize the overall business performance using the provided metrics.

### Most Important Business Finding
Identify the most important finding supported by the data.

### Potential Business Risk
Explain a risk that is directly supported by the provided metrics.
Do not introduce external assumptions.

### Actionable Recommendations
Provide exactly two actionable recommendations based ONLY on the observed data.

Recommendations must directly address the metrics or patterns provided above.

Allowed examples:
- Diversifying revenue when one product has very high revenue concentration.
- Increasing focus on lower-revenue products when revenue is highly concentrated.
- Reviewing product/category performance based on the provided revenue data.

Do NOT recommend:
- targeting customer segments
- entering new markets
- market research
- marketing campaigns
- customer surveys
- supplier changes
- inventory changes
- pricing changes
- geographic expansion

unless the provided business data explicitly contains evidence supporting
those recommendations.

Clearly present recommendations as suggestions, not as facts.

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
"""

    return prompt