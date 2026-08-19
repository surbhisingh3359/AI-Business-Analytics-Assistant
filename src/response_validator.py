from typing import Any


def validate_ai_response(
    response: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate an AI-generated business analysis against known business context.
    """

    if not isinstance(response, str):
        raise TypeError("response must be a string.")

    if not response.strip():
        raise ValueError("response cannot be empty.")

    if not isinstance(context, dict):
        raise TypeError("context must be a dictionary.")

    business_summary = context.get("business_summary", {})

    if not isinstance(business_summary, dict):
        raise ValueError("business_summary must be a dictionary.")

    warnings = []

    # ---------------------------------------------------------
    # 1. Validate important business metrics
    # ---------------------------------------------------------

    metrics_to_check = {
        "total_revenue": business_summary.get("total_revenue"),
        "total_orders": business_summary.get("total_orders"),
    }

    for metric_name, value in metrics_to_check.items():
        if value is None:
            continue

        if isinstance(value, float) and value.is_integer():
            formatted_values = {
                str(int(value)),
                f"{int(value):,}",
            }
        else:
            formatted_values = {
                str(value),
                f"{value:,.2f}",
            }

        if not any(
            formatted_value in response
            for formatted_value in formatted_values
        ):
            warnings.append(
                f"AI response does not mention {metric_name}."
            )

    # ---------------------------------------------------------
    # 2. Detect unsupported business claims
    # ---------------------------------------------------------

    unsupported_claims = [
    "market penetration",
    "customer interest",
    "customer demand",
    "market traction",
    "customer demographics",
    "customer preferences",
    "supplier performance",
    "supplier reliability",
    "geographic performance",
    "geographical performance",
    "decline in demand",
    "increase in demand",
    "demand is increasing",
    "demand is decreasing",
    "quality issues",
    "market saturation",
    "market is saturated",
    "market resilience",
    "target market segments",
    "emerging trends",
    "competitive pressure",
    "competition",
]

    response_lower = response.lower()

    for phrase in unsupported_claims:
        if phrase in response_lower:
            warnings.append(
                f"Unsupported business claim detected: '{phrase}'."
            )

    # ---------------------------------------------------------
    # 3. Detect overconfident statements
    # ---------------------------------------------------------

    risky_phrases = [
        "guaranteed",
        "definitely",
        "will certainly",
        "always",
        "never",
    ]

    for phrase in risky_phrases:
        if phrase in response_lower:
            warnings.append(
                f"Potentially overconfident statement detected: '{phrase}'."
            )

    # ---------------------------------------------------------
    # 4. Check for recommendations
    # ---------------------------------------------------------

    recommendation_keywords = [
        "recommend",
        "recommendation",
        "should",
        "suggest",
        "action",
        "strategy",
    ]

    has_recommendations = any(
        keyword in response_lower
        for keyword in recommendation_keywords
    )

    return {
        "is_valid": len(warnings) == 0,
        "has_recommendations": has_recommendations,
        "warnings": warnings,
    }

