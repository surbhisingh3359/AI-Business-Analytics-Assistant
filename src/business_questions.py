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

    This function is dataset-agnostic and supports the insight
    types produced by the analytics and insight layers.
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

    if insight_type in {
        "group_comparison",
        "group_difference",
    }:

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
            "What factors are driving this trend, and what "
            "business action should be considered?"
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
    # Anomaly
    # ---------------------------------------------------------

    if insight_type == "anomaly":

        return (
            "What could be causing this unusual pattern, "
            "and what action should the business consider?"
        )

    # ---------------------------------------------------------
    # Risk
    # ---------------------------------------------------------

    if insight_type == "risk":

        return (
            "What could be causing this risk, and what action "
            "should the business consider?"
        )

    # ---------------------------------------------------------
    # Concentration
    # ---------------------------------------------------------

    if insight_type in {
        "concentration",
        "product_concentration",
        "category_concentration",
    }:

        return (
            "How concentrated is this metric across the available "
            "products or categories, and what action should be considered?"
        )

    # ---------------------------------------------------------
    # Data quality
    # ---------------------------------------------------------

    if insight_type in {
        "data_quality",
        "quality",
    }:

        return (
            "What is causing this data-quality issue, and how "
            "could the reliability of the analysis be improved?"
        )

    # ---------------------------------------------------------
    # General insight
    # ---------------------------------------------------------

    return (
        "Why is this pattern occurring, and what business "
        "action should be considered?"
    )


def _get_column_names(
    context: dict[str, Any],
) -> list[str]:
    """
    Safely extract detected column names from dataset context.

    Supports both:
        context["dataset"]["column_names"]

    and older contexts where the dataset section may be absent.
    """

    dataset = context.get(
        "dataset",
        {},
    )

    if not isinstance(dataset, dict):
        return []

    columns = dataset.get(
        "column_names",
        [],
    )

    if not isinstance(columns, list):
        return []

    return [
        str(column)
        for column in columns
        if str(column).strip()
    ]


def _get_semantic_roles(
    context: dict[str, Any],
) -> dict[str, Any]:
    """Safely retrieve semantic roles from dataset context."""

    roles = context.get(
        "semantic_roles",
        {},
    )

    if not isinstance(roles, dict):
        return {}

    return roles


def _schema_question_candidates(
    context: dict[str, Any],
) -> list[str]:
    """
    Generate general analytical questions from detected schema.

    These questions do not assume that the dataset is a sales
    dataset. They are based only on detected columns and roles.
    """

    roles = _get_semantic_roles(
        context
    )

    columns = _get_column_names(
        context
    )

    questions: list[str] = []

    # ---------------------------------------------------------
    # Numeric analysis
    # ---------------------------------------------------------

    numeric_columns = roles.get(
        "numeric",
        [],
    )

    if isinstance(
        numeric_columns,
        list,
    ):

        for column in numeric_columns[:3]:

            column_name = str(
                column
            )

            questions.append(
                f"What are the key patterns in {column_name}?"
            )

    # ---------------------------------------------------------
    # Categorical / group analysis
    # ---------------------------------------------------------

    categorical_columns = roles.get(
        "categorical",
        [],
    )

    if isinstance(
        categorical_columns,
        list,
    ):

        for column in categorical_columns[:2]:

            column_name = str(
                column
            )

            questions.append(
                f"How does performance differ across "
                f"{column_name} groups?"
            )

    # ---------------------------------------------------------
    # Date/time analysis
    # ---------------------------------------------------------

    datetime_columns = roles.get(
        "datetime",
        [],
    )

    if isinstance(
        datetime_columns,
        list,
    ) and datetime_columns:

        questions.append(
            "How does the observed data change over time?"
        )

    # ---------------------------------------------------------
    # Relationship analysis
    # ---------------------------------------------------------

    if isinstance(
        numeric_columns,
        list,
    ) and len(numeric_columns) >= 2:

        questions.append(
            "Which numerical variables have meaningful relationships?"
        )

    # ---------------------------------------------------------
    # Dataset-level question
    # ---------------------------------------------------------

    if columns:

        questions.append(
            "What are the most important patterns and "
            "business opportunities supported by this dataset?"
        )

    return questions


def generate_business_questions(
    insights: list[dict[str, Any]],
    max_questions: int = 5,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate business-oriented questions from analytical insights.

    The function remains backward-compatible with the existing API:

        generate_business_questions(insights)

    An optional dataset context can also be supplied:

        generate_business_questions(
            insights,
            context=context,
        )

    When context is available, additional questions can be
    generated from detected dataset structure without assuming
    that the dataset belongs to a specific business domain.
    """

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    if not isinstance(
        insights,
        list,
    ):
        raise TypeError(
            "insights must be a list."
        )

    if not isinstance(
        max_questions,
        int,
    ):
        raise TypeError(
            "max_questions must be an integer."
        )

    if max_questions <= 0:
        raise ValueError(
            "max_questions must be greater than zero."
        )

    if context is not None and not isinstance(
        context,
        dict,
    ):
        raise TypeError(
            "context must be a dictionary."
        )

    # ---------------------------------------------------------
    # Generate questions
    # ---------------------------------------------------------

    results: list[
        dict[str, Any]
    ] = []

    selected_questions: set[str] = set()

    # ---------------------------------------------------------
    # 1. Insight-driven questions
    #
    # Preserve the existing behavior first because these
    # questions are directly grounded in detected insights.
    # ---------------------------------------------------------

    for insight in insights:

        if not isinstance(
            insight,
            dict,
        ):
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
                    insight.get(
                        "finding",
                        "",
                    ),
                ),
                "type": insight.get(
                    "type",
                    "general",
                ),
            }
        )

        if len(results) >= max_questions:
            return results

    # ---------------------------------------------------------
    # 2. Schema-driven questions
    #
    # Only used when dataset context is supplied.
    # ---------------------------------------------------------

    if context is not None:

        schema_questions = _schema_question_candidates(
            context
        )

        for question in schema_questions:

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
                    "source_insight": "",
                    "type": "schema_analysis",
                }
            )

            if len(results) >= max_questions:
                break

    return results