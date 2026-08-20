import pytest

from src.business_questions import generate_business_questions


def test_generates_business_question_from_insight():

    insights = [
        {
            "type": "group_comparison",
            "insight": (
                "IT has the highest average salary compared "
                "with other departments."
            ),
            "category": "department",
            "metric": "salary",
            "significance": 8,
            "impact": 8,
        }
    ]

    questions = generate_business_questions(insights)

    assert isinstance(questions, list)
    assert len(questions) > 0

    question = questions[0]

    assert "question" in question
    assert isinstance(question["question"], str)
    assert len(question["question"]) > 10


def test_questions_are_unique():

    insights = [
        {
            "type": "trend",
            "insight": "Revenue increased significantly over time.",
            "impact": 8,
        },
        {
            "type": "trend",
            "insight": "Revenue increased significantly over time.",
            "impact": 7,
        },
    ]

    questions = generate_business_questions(insights)

    question_texts = [
        item["question"].lower()
        for item in questions
    ]

    assert len(question_texts) == len(set(question_texts))


def test_respects_max_questions():

    insights = [
        {
            "type": "trend",
            "insight": f"Metric {i} increased significantly.",
            "impact": i,
        }
        for i in range(10)
    ]

    questions = generate_business_questions(
        insights,
        max_questions=3,
    )

    assert len(questions) <= 3


def test_empty_insights_returns_empty_list():

    assert generate_business_questions([]) == []


def test_invalid_input_is_rejected():

    with pytest.raises(TypeError):
        generate_business_questions("invalid")


def test_invalid_max_questions_is_rejected():

    with pytest.raises(ValueError):
        generate_business_questions(
            [],
            max_questions=0,
        )