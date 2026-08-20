from typing import Any


PRIORITY_SCORES = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}

CONFIDENCE_SCORES = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}


def _normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""

    return " ".join(
        str(text).lower().split()
    )


def _text_similarity(
    first: str,
    second: str,
) -> float:
    """Calculate simple Jaccard similarity between two texts."""

    first_words = set(
        _normalize_text(first).split()
    )

    second_words = set(
        _normalize_text(second).split()
    )

    if not first_words or not second_words:
        return 0.0

    return len(
        first_words & second_words
    ) / len(
        first_words | second_words
    )


def _is_duplicate(
    insight: str,
    selected: list[str],
) -> bool:
    """Detect exact or near-duplicate insights."""

    normalized = _normalize_text(insight)

    for existing in selected:

        existing_normalized = _normalize_text(
            existing
        )

        if normalized == existing_normalized:
            return True

        if (
            _text_similarity(
                normalized,
                existing_normalized,
            )
            >= 0.80
        ):
            return True

    return False


def _priority_score(
    priority: str,
) -> float:
    """Convert priority into a numeric score."""

    return PRIORITY_SCORES.get(
        str(priority).lower(),
        1.0,
    )


def _confidence_score(
    confidence: str,
) -> float:
    """Convert confidence into a numeric score."""

    return CONFIDENCE_SCORES.get(
        str(confidence).lower(),
        1.0,
    )


def _type_bonus(
    insight_type: str,
) -> float:
    """Give additional weight to important insight types."""

    bonuses = {
        "risk": 3.0,
        "anomaly": 3.0,
        "trend": 2.5,
        "group_difference": 2.0,
        "correlation": 2.0,
        "category_concentration": 2.0,
        "data_quality": 1.5,
    }

    return bonuses.get(
        str(insight_type).lower(),
        0.0,
    )


def _score_insight(
    insight: dict[str, Any],
) -> float:
    """
    Calculate an importance score for an insight.

    Supports both the new Insight Engine format and
    the older scoring fields used by existing tests.
    """

    score = 0.0

    # ---------------------------------------------------------
    # New Insight Engine fields
    # ---------------------------------------------------------

    score += _priority_score(
        insight.get(
            "priority",
            "medium",
        )
    )

    score += _confidence_score(
        insight.get(
            "confidence",
            "medium",
        )
    )

    score += _type_bonus(
        insight.get(
            "type",
            "",
        )
    )

    # ---------------------------------------------------------
    # Backward-compatible numeric scoring
    # ---------------------------------------------------------

    for field in (
        "impact",
        "significance",
        "actionability",
    ):

        value = insight.get(
            field,
            0,
        )

        if isinstance(
            value,
            (int, float),
        ):
            score += float(value)

    return score


def _get_insight_text(
    insight: dict[str, Any],
) -> str:
    """
    Extract the human-readable insight text.

    New Insight Engine uses `finding`.
    Older code uses `insight`.
    """

    finding = insight.get(
        "finding"
    )

    if finding:
        return str(finding)

    old_insight = insight.get(
        "insight"
    )

    if old_insight:
        return str(old_insight)

    title = insight.get(
        "title"
    )

    if title:
        return str(title)

    return ""


def rank_insights(
    insights: list[dict[str, Any]],
    max_insights: int = 5,
) -> list[dict[str, Any]]:
    """
    Rank meaningful insights and remove repetitive findings.

    Works with the universal Insight Engine output.
    """

    if not isinstance(
        insights,
        list,
    ):
        raise TypeError(
            "insights must be a list."
        )

    if not isinstance(
        max_insights,
        int,
    ):
        raise TypeError(
            "max_insights must be an integer."
        )

    if max_insights <= 0:
        raise ValueError(
            "max_insights must be greater than zero."
        )

    valid_insights: list[
        dict[str, Any]
    ] = []

    # ---------------------------------------------------------
    # Validate and score
    # ---------------------------------------------------------

    for insight in insights:

        if not isinstance(
            insight,
            dict,
        ):
            continue

        text = _get_insight_text(
            insight
        )

        if not text:
            continue

        item = dict(insight)

        # Preserve the actual text in a consistent field.
        item["insight"] = text

        item["score"] = _score_insight(
            item
        )

        valid_insights.append(
            item
        )

    # ---------------------------------------------------------
    # Highest-value findings first
    # ---------------------------------------------------------

    valid_insights.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # ---------------------------------------------------------
    # Remove repetitive insights
    # ---------------------------------------------------------

    selected: list[
        dict[str, Any]
    ] = []

    selected_texts: list[str] = []

    for insight in valid_insights:

        text = insight["insight"]

        if _is_duplicate(
            text,
            selected_texts,
        ):
            continue

        selected.append(
            insight
        )

        selected_texts.append(
            text
        )

        if len(selected) >= max_insights:
            break

    return selected