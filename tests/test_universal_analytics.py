import pandas as pd
import pytest

from src.analysis_planner import build_analysis_plan
from src.dataset_classifier import detect_semantic_roles
from src.universal_analytics import analyze_dataset


def test_sales_dataset_analysis():

    df = pd.DataFrame(
        {
            "product": [
                "Laptop",
                "Mouse",
                "Keyboard",
                "Laptop",
            ],
            "category": [
                "Electronics",
                "Electronics",
                "Electronics",
                "Electronics",
            ],
            "quantity": [2, 5, 3, 1],
            "price": [55000, 800, 1500, 55000],
        }
    )

    roles = detect_semantic_roles(df)

    plan = build_analysis_plan(
        df,
        roles,
    )

    result = analyze_dataset(
        df,
        roles,
        plan,
    )

    assert result["dataset_overview"]["rows"] == 4
    assert "price" in result["numeric_summary"]
    assert "category" in result["categorical_summary"]


def test_student_dataset_analysis():

    df = pd.DataFrame(
        {
            "student_id": [101, 102, 103, 104],
            "age": [20, 21, 19, 22],
            "attendance_percentage": [
                85,
                91,
                78,
                95,
            ],
            "study_hours": [
                3.5,
                5.0,
                2.0,
                6.0,
            ],
            "gender": [
                "F",
                "M",
                "F",
                "M",
            ],
        }
    )

    roles = detect_semantic_roles(df)

    plan = build_analysis_plan(
        df,
        roles,
    )

    result = analyze_dataset(
        df,
        roles,
        plan,
    )

    assert "age" in result["numeric_summary"]
    assert "attendance_percentage" in result["numeric_summary"]
    assert "gender" in result["categorical_summary"]
    assert result["correlations"]


def test_employee_dataset_analysis():

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

    result = analyze_dataset(
        df,
        roles,
        plan,
    )

    assert "salary" in result["numeric_summary"]
    assert "age" in result["numeric_summary"]
    assert "joining_date" in result["datetime_summary"]
    assert "department" in result["categorical_summary"]


def test_missing_values_are_detected():

    df = pd.DataFrame(
        {
            "age": [20, 21, None, 25],
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                None,
            ],
        }
    )

    roles = detect_semantic_roles(df)

    plan = build_analysis_plan(
        df,
        roles,
    )

    result = analyze_dataset(
        df,
        roles,
        plan,
    )

    assert result["data_quality"]["missing_values"]["age"] == 1
    assert result["data_quality"]["missing_values"]["city"] == 1


def test_empty_dataset_is_rejected():

    df = pd.DataFrame()

    with pytest.raises(ValueError):

        analyze_dataset(
            df,
            {},
            {},
        )


def test_group_comparison_returns_meaningful_top_comparisons():

    df = pd.DataFrame(
        {
            "department": [
                "IT",
                "IT",
                "HR",
                "HR",
                "Finance",
                "Finance",
            ],
            "salary": [
                80000,
                85000,
                50000,
                52000,
                65000,
                68000,
            ],
            "age": [
                25,
                27,
                30,
                32,
                35,
                36,
            ],
        }
    )

    roles = detect_semantic_roles(df)

    plan = build_analysis_plan(
        df,
        roles,
    )

    result = analyze_dataset(
        df,
        roles,
        plan,
    )

    comparisons = result["group_comparisons"]

    assert isinstance(comparisons, list)
    assert len(comparisons) > 0

    comparison = comparisons[0]

    assert "category" in comparison
    assert "metric" in comparison
    assert "highest_group" in comparison
    assert "lowest_group" in comparison
    assert "percentage_gap" in comparison


def test_group_comparison_ignores_high_cardinality_columns():

    df = pd.DataFrame(
        {
            "customer": [
                "Customer_1",
                "Customer_2",
                "Customer_3",
                "Customer_4",
                "Customer_5",
                "Customer_6",
                "Customer_7",
                "Customer_8",
                "Customer_9",
                "Customer_10",
                "Customer_11",
            ],
            "department": [
                "IT",
                "IT",
                "HR",
                "HR",
                "IT",
                "HR",
                "Finance",
                "Finance",
                "IT",
                "HR",
                "Finance",
            ],
            "salary": [
                80000,
                85000,
                50000,
                52000,
                82000,
                51000,
                65000,
                68000,
                81000,
                53000,
                67000,
            ],
        }
    )

    roles = detect_semantic_roles(df)

    plan = build_analysis_plan(
        df,
        roles,
    )

    result = analyze_dataset(
        df,
        roles,
        plan,
    )

    comparisons = result["group_comparisons"]

    assert all(
        comparison["category"] != "customer"
        for comparison in comparisons
    )


def test_group_comparison_ignores_negligible_differences():

    df = pd.DataFrame(
        {
            "department": [
                "IT",
                "IT",
                "HR",
                "HR",
            ],
            "salary": [
                50000,
                50100,
                50050,
                50020,
            ],
        }
    )

    roles = detect_semantic_roles(df)

    plan = build_analysis_plan(
        df,
        roles,
    )

    result = analyze_dataset(
        df,
        roles,
        plan,
    )

    assert result["group_comparisons"] == []

def test_group_comparison_identifies_strongest_difference():

    df = pd.DataFrame(
        {
            "department": [
                "IT",
                "IT",
                "HR",
                "HR",
                "Finance",
                "Finance",
            ],
            "salary": [
                90000,
                95000,
                50000,
                52000,
                65000,
                68000,
            ],
        }
    )

    roles = {
        "department": "categorical",
        "salary": "monetary",
    }

    plan = {
        "analyses": [
            "group_comparison_analysis",
        ],
        "columns_by_role": {
            "categorical": ["department"],
            "monetary": ["salary"],
        },
    }

    result = analyze_dataset(
        df,
        roles,
        plan,
    )

    comparisons = result["group_comparisons"]

    assert len(comparisons) == 1

    comparison = comparisons[0]

    assert comparison["category"] == "department"
    assert comparison["metric"] == "salary"
    assert comparison["highest_group"] == "IT"
    assert comparison["lowest_group"] == "HR"
    assert comparison["significance"] == "high"


def test_group_comparison_ignores_small_differences():

    df = pd.DataFrame(
        {
            "department": [
                "IT",
                "IT",
                "HR",
                "HR",
            ],
            "salary": [
                50000,
                51000,
                50500,
                51500,
            ],
        }
    )

    roles = {
        "department": "categorical",
        "salary": "monetary",
    }

    plan = {
        "analyses": [
            "group_comparison_analysis",
        ],
        "columns_by_role": {
            "categorical": ["department"],
            "monetary": ["salary"],
        },
    }

    result = analyze_dataset(
        df,
        roles,
        plan,
    )

    assert result["group_comparisons"] == []