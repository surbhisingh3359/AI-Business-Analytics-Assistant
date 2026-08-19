import pytest

from src.response_validator import validate_ai_response


@pytest.fixture
def context():
    return {
        "business_summary": {
            "total_revenue": 136000.0,
            "total_orders": 5,
            "total_quantity": 13,
            "average_order_value": 27200.0,
        }
    }


def test_valid_response(context):
    response = (
        "The business generated 136,000 revenue from 5 orders. "
        "I recommend diversifying revenue sources."
    )

    result = validate_ai_response(response, context)

    assert result["is_valid"] is True
    assert result["has_recommendations"] is True
    assert result["warnings"] == []


def test_missing_revenue(context):
    response = (
        "The business generated revenue from 5 orders. "
        "I recommend reviewing product concentration."
    )

    result = validate_ai_response(response, context)

    assert result["is_valid"] is False
    assert any(
        "total_revenue" in warning
        for warning in result["warnings"]
    )


def test_missing_orders(context):
    response = (
        "The business generated 136,000 in revenue. "
        "I recommend reviewing product concentration."
    )

    result = validate_ai_response(response, context)

    assert result["is_valid"] is False
    assert any(
        "total_orders" in warning
        for warning in result["warnings"]
    )


def test_overconfident_language(context):
    response = (
        "The business generated 136,000 revenue from 5 orders. "
        "Revenue will definitely increase next month."
    )

    result = validate_ai_response(response, context)

    assert result["is_valid"] is False
    assert any(
        "definitely" in warning
        for warning in result["warnings"]
    )


def test_recommendation_detection(context):
    response = (
        "The business generated 136,000 revenue from 5 orders. "
        "The company should diversify its products."
    )

    result = validate_ai_response(response, context)

    assert result["has_recommendations"] is True


def test_no_recommendations(context):
    response = (
        "The business generated 136,000 revenue from 5 orders."
    )

    result = validate_ai_response(response, context)

    assert result["has_recommendations"] is False


def test_invalid_response_type(context):
    with pytest.raises(TypeError):
        validate_ai_response(None, context)


def test_empty_response(context):
    with pytest.raises(ValueError):
        validate_ai_response("", context)


def test_invalid_context_type():
    with pytest.raises(TypeError):
        validate_ai_response("136000 revenue from 5 orders", None)


def test_detects_unsupported_market_claims():
    response = """
    Revenue is highly concentrated in the Laptop product.
    The business has limited market penetration and low customer interest.
    """

    context = {
        "business_summary": {
            "total_revenue": 136000.0,
            "total_orders": 5,
        }
    }

    result = validate_ai_response(response, context)

    assert result["is_valid"] is False
    assert any(
        "unsupported" in warning.lower()
        for warning in result["warnings"]
    )


def test_allows_data_grounded_business_statement():
    response = """
    Total revenue is 136000 and total orders are 5.
    The Laptop product generates most of the revenue.
    The business should diversify its product portfolio.
    """

    context = {
        "business_summary": {
            "total_revenue": 136000.0,
            "total_orders": 5,
        }
    }

    result = validate_ai_response(response, context)

    assert result["is_valid"] is True