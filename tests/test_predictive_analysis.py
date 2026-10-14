import pandas as pd
import pytest

from src.predictive_analysis import (
    build_model,
    detect_prediction_targets,
    detect_prediction_task,
    prepare_features,
    run_prediction,
    split_data,
    train_predictive_model,
)

# =========================================================
# TARGET DETECTION
# =========================================================

def test_detect_prediction_targets():

    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
            ],
            "quantity": [
                2,
                4,
                3,
                5,
                2,
            ],
            "price": [
                100,
                200,
                150,
                250,
                120,
            ],
        }
    )

    result = detect_prediction_targets(
        dataframe
    )

    assert isinstance(
        result,
        dict,
    )

    assert "regression_candidates" in result
    assert "classification_candidates" in result

    assert "price" in (
        result["regression_candidates"]
    )

    assert "quantity" in (
        result["regression_candidates"]
    )

    assert "category" in (
        result["classification_candidates"]
    )


# =========================================================
# INVALID DATAFRAME
# =========================================================

def test_predictive_analysis_rejects_invalid_dataframe():

    with pytest.raises(TypeError):

        detect_prediction_targets(
            "invalid"
        )


# =========================================================
# EMPTY DATAFRAME
# =========================================================

def test_predictive_analysis_rejects_empty_dataframe():

    with pytest.raises(ValueError):

        detect_prediction_targets(
            pd.DataFrame()
        )


# =========================================================
# TASK DETECTION — REGRESSION
# =========================================================

def test_detect_regression_task():

    dataframe = pd.DataFrame(
        {
            "price": [
                100,
                200,
                300,
                400,
            ]
        }
    )

    result = detect_prediction_task(
        dataframe,
        "price",
    )

    assert result == "regression"


# =========================================================
# TASK DETECTION — CLASSIFICATION
# =========================================================

def test_detect_classification_task():

    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "A",
                "B",
            ]
        }
    )

    result = detect_prediction_task(
        dataframe,
        "category",
    )

    assert result == "classification"


# =========================================================
# MISSING TARGET
# =========================================================

def test_detect_prediction_task_requires_target():

    dataframe = pd.DataFrame(
        {
            "price": [
                100,
                200,
            ]
        }
    )

    with pytest.raises(ValueError):

        detect_prediction_task(
            dataframe,
            "missing_column",
        )


# =========================================================
# INVALID TARGET TYPE
# =========================================================

def test_detect_prediction_task_requires_string_target():

    dataframe = pd.DataFrame(
        {
            "price": [
                100,
                200,
            ]
        }
    )

    with pytest.raises(TypeError):

        detect_prediction_task(
            dataframe,
            123,
        )


# =========================================================
# EMPTY TARGET
# =========================================================

def test_detect_prediction_task_rejects_empty_target():

    dataframe = pd.DataFrame(
        {
            "price": [
                None,
                None,
                None,
            ]
        }
    )

    with pytest.raises(ValueError):

        detect_prediction_task(
            dataframe,
            "price",
        )


# =========================================================
# FEATURE PREPARATION
# =========================================================

def test_prepare_features():

    dataframe = pd.DataFrame(
        {
            "order_id": [
                1,
                2,
                3,
                4,
                5,
            ],
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
            ],
            "quantity": [
                2,
                4,
                3,
                5,
                2,
            ],
            "price": [
                100,
                200,
                150,
                250,
                120,
            ],
        }
    )

    (
        X,
        y,
        numeric_features,
        categorical_features,
    ) = prepare_features(
        dataframe,
        "price",
    )

    assert isinstance(
        X,
        pd.DataFrame,
    )

    assert isinstance(
        y,
        pd.Series,
    )

    # Identifier should not be used as a feature.
    assert "order_id" not in X.columns

    assert "quantity" in numeric_features

    assert "category" in categorical_features

    assert len(X) == 5
    assert len(y) == 5


# =========================================================
# FEATURE PREPARATION — TOO LITTLE DATA
# =========================================================

def test_prepare_features_requires_enough_rows():

    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
            ],
            "price": [
                100,
                200,
            ],
        }
    )

    with pytest.raises(ValueError):

        prepare_features(
            dataframe,
            "price",
        )


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

def test_split_data():

    X = pd.DataFrame(
        {
            "feature_1": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
        }
    )

    y = pd.Series(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
        ],
        name="target",
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
        task="regression",
    )

    assert len(X_train) == 8
    assert len(X_test) == 2

    assert len(y_train) == 8
    assert len(y_test) == 2


# =========================================================
# MODEL CREATION
# =========================================================

def test_build_regression_model():

    model = build_model(
        "regression"
    )

    assert model.__class__.__name__ == (
        "LinearRegression"
    )


def test_build_classification_model():

    model = build_model(
        "classification"
    )

    assert model.__class__.__name__ == (
        "LogisticRegression"
    )


def test_build_model_rejects_unknown_task():

    with pytest.raises(ValueError):

        build_model(
            "unknown"
        )


# =========================================================
# REGRESSION MODEL
# =========================================================

def test_train_predictive_model_regression():

    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
            "quantity": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "price": [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
            ],
        }
    )

    result = train_predictive_model(
        dataframe,
        "price",
    )

    assert isinstance(
        result,
        dict,
    )

    assert result["status"] == "completed"

    assert result["task"] == "regression"

    assert result["target"] == "price"

    assert result["model_name"] == (
        "LinearRegression"
    )

    assert "metrics" in result

    assert "mae" in result["metrics"]

    assert "rmse" in result["metrics"]

    assert "r2" in result["metrics"]

    assert result["train_rows"] > 0

    assert result["test_rows"] > 0

    assert len(
        result["predictions"]
    ) == result["test_rows"]

    assert len(
        result["actual_values"]
    ) == result["test_rows"]


# =========================================================
# CLASSIFICATION MODEL
# =========================================================

def test_train_predictive_model_classification():

    dataframe = pd.DataFrame(
        {
            "age": [
                20,
                21,
                22,
                23,
                30,
                31,
                32,
                33,
                40,
                41,
                42,
                43,
            ],
            "income": [
                20000,
                21000,
                22000,
                23000,
                50000,
                51000,
                52000,
                53000,
                80000,
                81000,
                82000,
                83000,
            ],
            "segment": [
                "Low",
                "Low",
                "Low",
                "Low",
                "Medium",
                "Medium",
                "Medium",
                "Medium",
                "High",
                "High",
                "High",
                "High",
            ],
        }
    )

    result = train_predictive_model(
        dataframe,
        "segment",
    )

    assert isinstance(
        result,
        dict,
    )

    assert result["status"] == "completed"

    assert result["task"] == "classification"

    assert result["target"] == "segment"

    assert result["model_name"] == (
        "LogisticRegression"
    )

    assert "metrics" in result

    assert "accuracy" in result["metrics"]

    assert "precision" in result["metrics"]

    assert "recall" in result["metrics"]

    assert "f1" in result["metrics"]

    assert result["train_rows"] > 0

    assert result["test_rows"] > 0


# =========================================================
# UNKNOWN / INVALID TARGET
# =========================================================

def test_train_predictive_model_requires_valid_target():

    dataframe = pd.DataFrame(
        {
            "price": [
                100,
                200,
                300,
                400,
                500,
            ]
        }
    )

    with pytest.raises(ValueError):

        train_predictive_model(
            dataframe,
            "missing",
        )

# =========================================================
# PUBLIC PREDICTION API
# =========================================================

def test_run_prediction_regression():

    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
            "quantity": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "price": [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
            ],
        }
    )

    result = run_prediction(
        dataframe,
        "price",
    )

    assert result["status"] == "completed"

    assert result["task"] == "regression"

    assert result["target"] == "price"

    assert "model" not in result

    assert "metrics" in result

    assert "predictions" in result

    assert "actual_values" in result


def test_run_prediction_classification():

    dataframe = pd.DataFrame(
        {
            "age": [
                20,
                21,
                22,
                23,
                30,
                31,
                32,
                33,
                40,
                41,
                42,
                43,
            ],
            "income": [
                20000,
                21000,
                22000,
                23000,
                50000,
                51000,
                52000,
                53000,
                80000,
                81000,
                82000,
                83000,
            ],
            "segment": [
                "Low",
                "Low",
                "Low",
                "Low",
                "Medium",
                "Medium",
                "Medium",
                "Medium",
                "High",
                "High",
                "High",
                "High",
            ],
        }
    )

    result = run_prediction(
        dataframe,
        "segment",
    )

    assert result["status"] == "completed"

    assert result["task"] == "classification"

    assert result["target"] == "segment"

    assert "model" not in result

    assert "metrics" in result

    assert "accuracy" in result["metrics"]

    assert "precision" in result["metrics"]

    assert "recall" in result["metrics"]

    assert "f1" in result["metrics"]

    assert result["train_rows"] > 0

    assert result["test_rows"] > 0

def test_run_prediction_rejects_invalid_target():

    dataframe = pd.DataFrame(
        {
            "price": [
                100,
                200,
                300,
                400,
                500,
            ]
        }
    )

    with pytest.raises(ValueError):

        run_prediction(
            dataframe,
            "unknown_target",
        )

def test_prediction_targets_exclude_identifier_columns():

    dataframe = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "customer_id": [10, 11, 12, 13, 14],
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
            ],
            "quantity": [1, 2, 3, 4, 5],
            "price": [100, 200, 300, 400, 500],
        }
    )

    result = detect_prediction_targets(
        dataframe
    )

    assert "order_id" not in (
        result["regression_candidates"]
    )

    assert "customer_id" not in (
        result["regression_candidates"]
    )

    assert "quantity" in (
        result["regression_candidates"]
    )

    assert "price" in (
        result["regression_candidates"]
    )

    assert "category" in (
        result["classification_candidates"]
    )