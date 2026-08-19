import pytest

from src.prompt_builder import build_business_prompt


def test_build_business_prompt():
    context = {
        "business_summary": {
            "total_revenue": 136000.0,
            "total_orders": 5,
            "total_quantity": 13,
            "average_order_value": 27200.0,
        },
        "business_insights": {
            "top_category": {
                "category": "Electronics",
                "revenue": 118500.0,
                "contribution_percentage": 87.13,
            },
            "top_product": {
                "product": "Laptop",
                "revenue": 110000.0,
                "concentration_percentage": 80.88,
                "risk_level": "high",
            },
            "product_concentration_insight": (
                "High Product Concentration: Laptop generates "
                "₹110,000.00, representing 80.88% of total revenue."
            ),
        },
    }

    prompt = build_business_prompt(context)

    assert "expert business analyst" in prompt
    assert "136,000.00" in prompt
    assert "Electronics" in prompt
    assert "118,500.00" in prompt
    assert "Laptop" in prompt
    assert "80.88%" in prompt
    assert "high" in prompt
    assert "actionable recommendations" in prompt


def test_build_business_prompt_invalid_input():
    with pytest.raises(TypeError):
        build_business_prompt("not a dictionary")


def test_build_business_prompt_missing_summary():
    context = {
        "business_insights": {}
    }

    with pytest.raises(ValueError):
        build_business_prompt(context)


def test_build_business_prompt_missing_insights():
    context = {
        "business_summary": {}
    }

    with pytest.raises(ValueError):
        build_business_prompt(context)