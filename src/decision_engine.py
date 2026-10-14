from typing import Any


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def _normalize_text(text: str) -> str:
    """Normalize text for consistent comparisons."""

    return " ".join(
        str(text).lower().split()
    )


# =========================================================
# INSIGHT HELPERS
# =========================================================

def _get_insight_text(
    insight: dict[str, Any],
) -> str:
    """Extract the human-readable insight text."""

    for field in (
        "insight",
        "finding",
        "title",
    ):
        value = insight.get(field)

        if value:
            return str(value).strip()

    return ""


def _get_insight_type(
    insight: dict[str, Any],
) -> str:
    """Return a normalized insight type."""

    return str(
        insight.get(
            "type",
            "general",
        )
    ).strip().lower()


# =========================================================
# RECOMMENDED ANALYSIS
# =========================================================

def _recommended_analysis(
    insight_type: str,
) -> str:
    """
    Map an insight type to the appropriate
    next analysis.
    """

    mapping = {
        "group_comparison": "group_comparison",
        "group_difference": "group_comparison",

        "correlation": "correlation_analysis",

        # Important:
        # Existing tests expect time_analysis.
        "trend": "time_analysis",

        "anomaly": "anomaly_analysis",

        "risk": "risk_analysis",

        "category_concentration": "group_comparison",

        "data_quality": "data_quality_analysis",
    }

    return mapping.get(
        insight_type,
        "descriptive_analysis",
    )


# =========================================================
# DECISION QUESTION
# =========================================================

def _build_decision_question(
    insight: dict[str, Any],
) -> str:
    """Convert an insight into a decision-oriented question."""

    insight_type = _get_insight_type(
        insight
    )

    insight_text = _get_insight_text(
        insight
    )

    # -----------------------------------------------------
    # Group comparison
    # -----------------------------------------------------

    if insight_type in {
        "group_comparison",
        "group_difference",
    }:

        metric = insight.get(
            "metric"
        )

        category = insight.get(
            "category"
        )

        if metric and category:
            return (
                f"Should the business investigate "
                f"the differences in {metric} "
                f"across {category}?"
            )

        if metric:
            return (
                f"Should the business investigate "
                f"the differences in {metric}?"
            )

        return (
            "Should the business investigate "
            "the observed differences between groups?"
        )

    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    if insight_type == "correlation":

        return (
            "Should the business investigate "
            "the observed relationship between "
            "the variables?"
        )

    # -----------------------------------------------------
    # Trend
    # -----------------------------------------------------

    if insight_type == "trend":

        return (
            "Should the business investigate "
            "the observed trend over time?"
        )

    # -----------------------------------------------------
    # Anomaly
    # -----------------------------------------------------

    if insight_type == "anomaly":

        return (
            "Should the business investigate "
            "the detected anomaly?"
        )

    # -----------------------------------------------------
    # Risk
    # -----------------------------------------------------

    if insight_type == "risk":

        # Include the actual insight in the decision.
        #
        # This is important because multiple risk insights
        # must remain distinct:
        #
        # Insight 1 -> decision about Insight 1
        # Insight 2 -> decision about Insight 2
        #
        # while identical insights still get removed by
        # the duplicate logic in build_decision_layer().
        return (
            "Should the business investigate "
            f"the identified potential risk described by: "
            f"{insight_text}?"
        )

    # -----------------------------------------------------
    # Category concentration
    # -----------------------------------------------------

    if insight_type == "category_concentration":

        return (
            "Should the business investigate "
            "the concentration of the observed metric?"
        )

    # -----------------------------------------------------
    # Data quality
    # -----------------------------------------------------

    if insight_type == "data_quality":

        return (
            "Should the business address the detected "
            "data-quality issue before making decisions?"
        )

    # -----------------------------------------------------
    # General insight
    # -----------------------------------------------------

    return (
        "Should the business investigate the observed "
        f"pattern described by: {insight_text}?"
    )


# =========================================================
# LIMITATIONS
# =========================================================

def _build_limitation(
    insight: dict[str, Any],
) -> str:
    """Describe what cannot safely be concluded."""

    insight_type = _get_insight_type(
        insight
    )

    # -----------------------------------------------------
    # Correlation
    # -----------------------------------------------------

    if insight_type == "correlation":

        return (
            "Correlation does not establish causation."
        )

    # -----------------------------------------------------
    # Trend
    # -----------------------------------------------------

    if insight_type == "trend":

        return (
            "The observed trend does not establish "
            "that the pattern will continue."
        )

    # -----------------------------------------------------
    # Group comparison
    # -----------------------------------------------------

    if insight_type in {
        "group_comparison",
        "group_difference",
        "category_concentration",
    }:

        return (
            "The observed difference or concentration "
            "does not by itself explain its underlying cause."
        )

    # -----------------------------------------------------
    # Anomaly / Risk
    # -----------------------------------------------------

    if insight_type in {
        "anomaly",
        "risk",
    }:

        return (
            "The observed pattern identifies a potential "
            "issue but does not establish its root cause."
        )

    # -----------------------------------------------------
    # General
    # -----------------------------------------------------

    return (
        "The available data supports the observed pattern "
        "but does not establish external causes."
    )


# =========================================================
# BUILD SINGLE DECISION
# =========================================================

def build_decision(
    insight: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert one analytical insight into a grounded
    decision-support object.
    """

    if not isinstance(
        insight,
        dict,
    ):
        raise TypeError(
            "insight must be a dictionary."
        )

    insight_text = _get_insight_text(
        insight
    )

    if not insight_text:
        raise ValueError(
            "insight must contain readable insight text."
        )

    insight_type = _get_insight_type(
        insight
    )

    decision = _build_decision_question(
        insight
    )

    recommended_analysis = _recommended_analysis(
        insight_type
    )

    limitations = [
        _build_limitation(
            insight
        )
    ]

    confidence = str(
        insight.get(
            "confidence",
            "medium",
        )
    ).lower()

    if confidence not in {
        "high",
        "medium",
        "low",
    }:
        confidence = "medium"

    return {
        "decision": decision,

        "evidence": [
            insight_text
        ],

        "recommended_analysis": (
            recommended_analysis
        ),

        "metrics": (
            [str(insight["metric"])]
            if insight.get("metric")
            else []
        ),

        "dimensions": (
            [str(insight["category"])]
            if insight.get("category")
            else []
        ),

        "confidence": confidence,

        "limitations": limitations,

        "source_type": insight_type,
    }


# =========================================================
# DECISION DUPLICATE KEY
# =========================================================

def _decision_duplicate_key(
    decision: dict[str, Any],
) -> tuple[str, str, tuple[str, ...]]:
    """
    Build a stable identity for a decision.

    We deliberately include evidence.

    Why?

    Two different insights can produce the same generic
    decision question.

    Example:

        Insight 0
        Insight 1
        Insight 2

    All are risk insights.

    If we only compare the decision text, they look
    identical.

    Including evidence means:

        same decision + same evidence
            -> duplicate

        same decision + different evidence
            -> different decision
    """

    normalized_decision = _normalize_text(
        decision.get(
            "decision",
            "",
        )
    )

    source_type = _normalize_text(
        decision.get(
            "source_type",
            "",
        )
    )

    evidence = decision.get(
        "evidence",
        [],
    )

    if not isinstance(
        evidence,
        list,
    ):
        evidence = [
            evidence
        ]

    normalized_evidence = tuple(
        _normalize_text(item)
        for item in evidence
    )

    return (
        normalized_decision,
        source_type,
        normalized_evidence,
    )


# =========================================================
# BUILD DECISION LAYER
# =========================================================

def build_decision_layer(
    insights: list[dict[str, Any]],
    max_decisions: int = 5,
) -> list[dict[str, Any]]:
    """
    Build grounded decision-support objects from insights.

    Rules:

    1. Input must be a list.
    2. max_decisions must be a positive integer.
    3. Invalid insights are skipped.
    4. Identical insights are removed.
    5. Different insights are preserved.
    6. Results are limited to max_decisions.
    """

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not isinstance(
        insights,
        list,
    ):
        raise TypeError(
            "insights must be a list."
        )

    if not isinstance(
        max_decisions,
        int,
    ):
        raise TypeError(
            "max_decisions must be an integer."
        )

    if max_decisions <= 0:
        raise ValueError(
            "max_decisions must be greater than zero."
        )

    # -----------------------------------------------------
    # Result containers
    # -----------------------------------------------------

    results: list[
        dict[str, Any]
    ] = []

    selected_decisions: set[
        tuple[str, str, tuple[str, ...]]
    ] = set()

    # -----------------------------------------------------
    # Convert insights -> decisions
    # -----------------------------------------------------

    for insight in insights:

        if not isinstance(
            insight,
            dict,
        ):
            continue

        try:
            decision = build_decision(
                insight
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        # -------------------------------------------------
        # Duplicate detection
        # -------------------------------------------------

        duplicate_key = (
            _decision_duplicate_key(
                decision
            )
        )

        if duplicate_key in selected_decisions:
            continue

        selected_decisions.add(
            duplicate_key
        )

        # -------------------------------------------------
        # Add result
        # -------------------------------------------------

        results.append(
            decision
        )

        # -------------------------------------------------
        # Respect requested limit
        # -------------------------------------------------

        if len(results) >= max_decisions:
            break

    return results