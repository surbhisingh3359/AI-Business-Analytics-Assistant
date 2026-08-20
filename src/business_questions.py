from typing import Any


def _normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""

    return " ".join(
        str(text).lower().split()
    )


def _question_from_insight(
    insight: dict[str, Any],
) -> str:
    """
    Convert an analytical insight into a business-oriented question.

    The function is intentionally dataset-agnostic.
    """

    insight_text = str(
        insight.get("insight", "")
    ).strip()

    insight_type = str(
        insight.get("type", "")
    ).lower()

    category = str(
        insight.get("category", "")
    ).strip()

    metric = str(
        insight.get("metric", "")
    ).strip()

    if not insight_text:
        return ""

    # ---------------------------------------------------------
    # Group comparison
    # ---------------------------------------------------------

    if insight_type == "group_comparison":

        if category and metric:
            return (
                f"What factors may explain the difference in "
                f"{metric} across {category} groups?"
            )

        return (
            "What factors may explain the difference between "
            "the observed groups?"
        )

    # ---------------------------------------------------------
    # Trend
    # ---------------------------------------------------------

    if insight_type == "trend":

        return (
            "What factors are driving this trend, and is it "
            "likely to continue?"
        )

    # ---------------------------------------------------------
    # Correlation
    # ---------------------------------------------------------

    if insight_type == "correlation":

        return (
            "What factors could explain this relationship, "
            "and how useful is it for business decision-making?"
        )

    # ---------------------------------------------------------
    # Anomaly / risk
    # ---------------------------------------------------------

    if insight_type in {
        "anomaly",
        "risk",
    }:

        return (
            "What could be causing this issue, and what action "
            "should the business consider?"
        )

    # ---------------------------------------------------------
    # General insight
    # ---------------------------------------------------------

    return (
        "Why is this pattern occurring, and what business "
        "action should be considered?"
    )


def generate_business_questions(
    insights: list[dict[str, Any]],
    max_questions: int = 5,
) -> list[dict[str, Any]]:
    """
    Generate business-oriented questions from ranked insights.

    Parameters
    ----------
    insights:
        List of analytical insight dictionaries.

    max_questions:
        Maximum number of questions to return.

    Returns
    -------
    list[dict[str, Any]]
        Business questions linked to their source insights.
    """

    if not isinstance(insights, list):
        raise TypeError(
            "insights must be a list."
        )

    if not isinstance(max_questions, int):
        raise TypeError(
            "max_questions must be an integer."
        )

    if max_questions <= 0:
        raise ValueError(
            "max_questions must be greater than zero."
        )

    results: list[dict[str, Any]] = []

    selected_questions: set[str] = set()

    for insight in insights:

        if not isinstance(insight, dict):
            continue

        question = _question_from_insight(
            insight
        )

        if not question:
            continue

        normalized_question = _normalize_text(
            question
        )

        if normalized_question in selected_questions:
            continue

        selected_questions.add(
            normalized_question
        )

        results.append(
            {
                "question": question,
                "source_insight": insight.get(
                    "insight",
                    "",
                ),
                "type": insight.get(
                    "type",
                    "general",
                ),
            }
        )

        if len(results) >= max_questions:
            break

    return results