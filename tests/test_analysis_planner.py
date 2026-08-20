import pandas as pd
import pytest

from src.analysis_planner import build_analysis_plan
from src.dataset_classifier import detect_semantic_roles


def test_sales_dataset_gets_business_analyses():

    df = pd.DataFrame(
        {
            "order_id": [1001, 1002, 1003],
            "product": ["Laptop", "Mouse", "Keyboard"],
            "category": [
                "Electronics",
                "Electronics",
                "Electronics",
            ],
            "quantity": [2, 5, 3],
            "price": [55000, 800, 1500],
        }
    )

    roles = detect_semantic_roles(df)

    plan = build_analysis_plan(
        df,
        roles,
    )

    assert "dataset_overview" in plan["analyses"]
    assert "data_quality" in plan["analyses"]
    assert "monetary_analysis" in plan["analyses"]
    assert "quantity_analysis" in plan["analyses"]
    assert "categorical_analysis" in plan["analyses"]


def test_student_dataset_gets_student_relevant_analysis():

    df = pd.DataFrame(
        {
            "student_id": [101, 102, 103, 104],
            "age": [20, 21, 19, 22],
            "attendance_percentage": [85, 91, 78, 95],
            "study_hours": [3.5, 5.0, 2.0, 6.0],
            "gender": ["F", "M", "F", "M"],
        }
    )

    roles = detect_semantic_roles(df)

    plan = build_analysis_plan(
        df,
        roles,
    )

    assert "age_analysis" in plan["analyses"]
    assert "percentage_analysis" in plan["analyses"]
    assert "correlation_analysis" in plan["analyses"]
    assert "group_comparison_analysis" in plan["analyses"]


def test_employee_dataset_gets_time_and_salary_analysis():

    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": [
                "IT",
                "HR",
                "IT",
                "Finance",
            ],
            "salary": [
                60000,
                50000,
                75000,
                65000,
            ],
            "age": [25, 31, 28, 35],
            "joining_date": [
                "2022-01-10",
                "2021-05-15",
                "2023-02-20",
                "2020-08-01",
            ],
        }
    )

    roles = detect_semantic_roles(df)

    plan = build_analysis_plan(
        df,
        roles,
    )

    assert "monetary_analysis" in plan["analyses"]
    assert "age_analysis" in plan["analyses"]
    assert "time_series_analysis" in plan["analyses"]
    assert "trend_analysis" in plan["analyses"]


def test_empty_dataset_is_rejected():

    df = pd.DataFrame()

    with pytest.raises(ValueError):
        build_analysis_plan(
            df,
            {},
        )