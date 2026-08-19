def build_business_prompt(context: dict) -> str:
    """Build a grounded AI prompt from structured business context."""

    if not isinstance(context, dict):
        raise TypeError("context must be a dictionary.")

    if "business_summary" not in context:
        raise ValueError("Missing business_summary in context.")

    if "business_insights" not in context:
        raise ValueError("Missing business_insights in context.")

    summary = context["business_summary"]
    insights = context["business_insights"]

    top_category = insights["top_category"]
    top_product = insights["top_product"]

    prompt = f"""
You are an expert business analyst.

Analyze ONLY the business data provided below.

Your job is to:
- Identify important patterns in the supplied data.
- Explain their business meaning.
- Identify risks supported by the data.
- Provide practical recommendations based on the available evidence.

IMPORTANT DATA-GROUNDING RULES
------------------------------
1. Do not invent facts, metrics, trends, customers, markets, suppliers,
   competitors, geography, or business conditions that are not provided.

2. Do not claim that demand is increasing or decreasing unless the data
   contains evidence of a trend.

3. Do not make claims about customer behavior unless customer data is provided.

4. Do not make claims about suppliers or supply-chain performance unless
   supplier data is provided.

5. Do not make claims about geographic or market performance unless geographic
   or market data is provided.

6. Recommendations must be directly connected to the supplied business data.

7. If a recommendation requires information that is not available, clearly
   label it as a general suggestion or hypothesis rather than a fact.

8. Do not present predictions as certainties.

9. Use the provided metrics as the source of truth. Do not recalculate or
   invent alternative values.

BUSINESS METRICS
----------------
Total Revenue: ₹{summary["total_revenue"]:,.2f}
Total Orders: {summary["total_orders"]}
Total Quantity Sold: {summary["total_quantity"]}
Average Order Value: ₹{summary["average_order_value"]:,.2f}

TOP CATEGORY
------------
Category: {top_category["category"]}
Revenue: ₹{top_category["revenue"]:,.2f}
Revenue Contribution: {top_category["contribution_percentage"]:.2f}%

TOP PRODUCT
-----------
Product: {top_product["product"]}
Revenue: ₹{top_product["revenue"]:,.2f}
Revenue Concentration: {top_product["concentration_percentage"]:.2f}%
Risk Level: {top_product["risk_level"]}

PRODUCT CONCENTRATION INSIGHT
-----------------------------
{insights["product_concentration_insight"]}

REQUIRED OUTPUT
---------------
Provide:

1. A short executive summary.
2. The most important business finding.
3. Potential business risks supported by the data.
4. Two actionable recommendations.

Keep the analysis concise, evidence-based, and suitable for a business
decision-maker.
"""

    return prompt.strip()