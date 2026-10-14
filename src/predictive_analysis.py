from typing import Any

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# CONFIGURATION
# =========================================================

MIN_PREDICTION_ROWS = 10
MAX_MODEL_ROWS = 5000
MAX_CATEGORICAL_LEVELS = 30
MAX_PREDICTION_POINTS = 500


# =========================================================
# VALIDATION
# =========================================================

def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the input dataset."""

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot run predictive analysis on an empty dataset."
        )


def _validate_target(
    dataframe: pd.DataFrame,
    target: str,
) -> None:
    """Validate a prediction target."""

    if not isinstance(
        target,
        str,
    ):
        raise TypeError(
            "target must be a string."
        )

    if target not in dataframe.columns:
        raise ValueError(
            f"Target column '{target}' does not exist."
        )

    if dataframe[target].dropna().empty:
        raise ValueError(
            "Target column contains no usable values."
        )


def _validate_test_size(
    test_size: float,
) -> None:
    """Validate the requested test proportion."""

    if not isinstance(
        test_size,
        (int, float),
    ):
        raise TypeError(
            "test_size must be numeric."
        )

    if not 0 < float(test_size) < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )


# =========================================================
# IDENTIFIER
# =========================================================

def _is_identifier_column(
    column: str,
) -> bool:
    """Identify common identifier-like column names."""

    name = str(
        column
    ).lower().strip()

    return (
        name == "id"
        or name.endswith("_id")
    )


# =========================================================
# READINESS
# =========================================================

def can_run_prediction(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Check whether predictive modelling is allowed.

    Small datasets can still expose candidate targets,
    but actual modelling requires at least 10 rows.
    """

    validate_dataframe(
        dataframe
    )

    usable_rows = int(
        len(dataframe)
    )

    if usable_rows < MIN_PREDICTION_ROWS:

        return {
            "available": False,
            "usable_rows": usable_rows,
            "minimum_rows": MIN_PREDICTION_ROWS,
            "reason": (
                f"At least {MIN_PREDICTION_ROWS} rows are "
                "required for predictive modeling and evaluation."
            ),
        }

    return {
        "available": True,
        "usable_rows": usable_rows,
        "minimum_rows": MIN_PREDICTION_ROWS,
        "reason": None,
    }


# =========================================================
# COLUMN DETECTION
# =========================================================

def get_numeric_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return numeric columns."""

    validate_dataframe(
        dataframe
    )

    return dataframe.select_dtypes(
        include="number"
    ).columns.tolist()


def get_categorical_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return categorical columns."""

    validate_dataframe(
        dataframe
    )

    columns: list[str] = []

    for column in dataframe.columns:

        series = dataframe[column]

        if (
            pd.api.types.is_object_dtype(
                series
            )
            or pd.api.types.is_string_dtype(
                series
            )
            or isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )
        ):
            columns.append(
                column
            )

    return columns


# =========================================================
# TARGET DETECTION
# =========================================================

def detect_prediction_targets(
    dataframe: pd.DataFrame,
) -> dict[str, list[str]]:
    """Detect usable prediction targets."""

    validate_dataframe(
        dataframe
    )

    numeric_columns = get_numeric_columns(
        dataframe
    )

    categorical_columns = get_categorical_columns(
        dataframe
    )

    regression_candidates = [
        column
        for column in numeric_columns
        if not _is_identifier_column(
            column
        )
    ]

    classification_candidates = [
        column
        for column in categorical_columns
        if not _is_identifier_column(
            column
        )
    ]

    return {
        "regression_candidates": regression_candidates,
        "classification_candidates": classification_candidates,
    }


def detect_prediction_task(
    dataframe: pd.DataFrame,
    target: str,
) -> str:
    """Detect regression or classification."""

    validate_dataframe(
        dataframe
    )

    _validate_target(
        dataframe,
        target,
    )

    series = dataframe[target].dropna()

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return "regression"

    return "classification"


# =========================================================
# FEATURE PREPARATION
# =========================================================

def prepare_features(
    dataframe: pd.DataFrame,
    target: str,
    max_categorical_levels: int = MAX_CATEGORICAL_LEVELS,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    list[str],
    list[str],
]:
    """
    Prepare a bounded feature set.

    Identifier columns and high-cardinality categorical
    columns are excluded from the baseline model.
    """

    validate_dataframe(
        dataframe
    )

    _validate_target(
        dataframe,
        target,
    )

    if not isinstance(
        max_categorical_levels,
        int,
    ):
        raise TypeError(
            "max_categorical_levels must be an integer."
        )

    if max_categorical_levels <= 1:
        raise ValueError(
            "max_categorical_levels must be greater than one."
        )

    model_data = dataframe.dropna(
        subset=[target]
    ).copy()

    if len(model_data) < 3:
        raise ValueError(
            "At least three usable rows are required "
            "for predictive modeling."
        )

    X = model_data.drop(
        columns=[target]
    )

    y = model_data[target]

    identifier_columns = [
        column
        for column in X.columns
        if _is_identifier_column(
            column
        )
    ]

    if identifier_columns:
        X = X.drop(
            columns=identifier_columns
        )

    numeric_features = (
        X.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_features: list[str] = []

    for column in X.columns:

        series = X[column]

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )
        ):

            unique_count = int(
                series.dropna().nunique()
            )

            if unique_count <= max_categorical_levels:
                categorical_features.append(
                    column
                )

    excluded_columns = [
        column
        for column in X.columns
        if (
            column not in numeric_features
            and column not in categorical_features
        )
    ]

    if excluded_columns:
        X = X.drop(
            columns=excluded_columns
        )

    if (
        not numeric_features
        and not categorical_features
    ):
        raise ValueError(
            "No usable predictive features were found."
        )

    return (
        X,
        y,
        numeric_features,
        categorical_features,
    )


# =========================================================
# PREPROCESSING
# =========================================================

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a bounded preprocessing pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split data safely for model validation."""

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if not isinstance(
        y,
        pd.Series,
    ):
        raise TypeError(
            "y must be a pandas Series."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    if len(X) < 5:
        raise ValueError(
            "At least 5 rows are required for train/test splitting."
        )

    _validate_test_size(
        test_size
    )

    test_rows = max(
        2,
        int(
            round(
                len(X)
                * float(test_size)
            )
        ),
    )

    stratify = None

    if task == "classification":

        class_counts = y.value_counts()

        number_of_classes = len(
            class_counts
        )

        if (
            number_of_classes > 1
            and class_counts.min() >= 2
        ):

            test_rows = max(
                test_rows,
                number_of_classes,
            )

            if (
                test_rows < len(X)
                and len(X) - test_rows
                >= number_of_classes
            ):
                stratify = y

    if test_rows >= len(X):

        raise ValueError(
            "The test set must contain fewer rows "
            "than the training set."
        )

    actual_test_size = (
        test_rows / len(X)
    )

    return train_test_split(
        X,
        y,
        test_size=actual_test_size,
        random_state=random_state,
        stratify=stratify,
    )


# =========================================================
# MODEL
# =========================================================

def build_model(
    task: str,
) -> Any:
    """Build a lightweight interpretable baseline model."""

    if task == "regression":
        return LinearRegression()

    if task == "classification":
        return LogisticRegression(
            max_iter=500,
            solver="lbfgs",
        )

    raise ValueError(
        f"Unsupported prediction task: {task}"
    )


# =========================================================
# BASELINE
# =========================================================

def evaluate_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
    task: str,
) -> dict[str, float]:
    """Evaluate a simple non-ML baseline."""

    if y_train.empty:
        raise ValueError(
            "y_train cannot be empty."
        )

    if y_test.empty:
        raise ValueError(
            "y_test cannot be empty."
        )

    if task == "regression":

        baseline_value = float(
            y_train.mean()
        )

        predictions = [
            baseline_value
            for _ in range(
                len(y_test)
            )
        ]

        mae = float(
            mean_absolute_error(
                y_test,
                predictions,
            )
        )

        mse = float(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        rmse = mse ** 0.5

        try:
            r2 = float(
                r2_score(
                    y_test,
                    predictions,
                )
            )
        except Exception:
            r2 = float("nan")

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    if task == "classification":

        majority_class = (
            y_train
            .value_counts()
            .idxmax()
        )

        predictions = [
            majority_class
            for _ in range(
                len(y_test)
            )
        ]

        return {
            "accuracy": float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

    raise ValueError(
        f"Unsupported prediction task: {task}"
    )


# =========================================================
# MODEL QUALITY
# =========================================================

def compare_model_with_baseline(
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    task: str,
) -> dict[str, Any]:
    """Compare model performance with the baseline."""

    if task == "regression":

        model_mae = float(
            model_metrics.get(
                "mae",
                float("inf"),
            )
        )

        baseline_mae = float(
            baseline_metrics.get(
                "mae",
                float("inf"),
            )
        )

        model_rmse = float(
            model_metrics.get(
                "rmse",
                float("inf"),
            )
        )

        baseline_rmse = float(
            baseline_metrics.get(
                "rmse",
                float("inf"),
            )
        )

        model_r2 = float(
            model_metrics.get(
                "r2",
                float("-inf"),
            )
        )

        baseline_r2 = float(
            baseline_metrics.get(
                "r2",
                float("-inf"),
            )
        )

        improves_mae = model_mae < baseline_mae
        improves_rmse = model_rmse < baseline_rmse
        improves_r2 = model_r2 > baseline_r2

        model_is_better = (
            (
                improves_mae
                and improves_rmse
            )
            or improves_r2
        )

        return {
            "baseline": baseline_metrics,
            "model": model_metrics,
            "mae_improvement": (
                baseline_mae
                - model_mae
            ),
            "rmse_improvement": (
                baseline_rmse
                - model_rmse
            ),
            "r2_improvement": (
                model_r2
                - baseline_r2
            ),
            "model_is_better": bool(
                model_is_better
            ),
            "quality": (
                "useful"
                if model_is_better
                else "needs_review"
            ),
        }

    if task == "classification":

        model_accuracy = float(
            model_metrics.get(
                "accuracy",
                0,
            )
        )

        baseline_accuracy = float(
            baseline_metrics.get(
                "accuracy",
                0,
            )
        )

        model_f1 = float(
            model_metrics.get(
                "f1",
                0,
            )
        )

        baseline_f1 = float(
            baseline_metrics.get(
                "f1",
                0,
            )
        )

        accuracy_improvement = (
            model_accuracy
            - baseline_accuracy
        )

        f1_improvement = (
            model_f1
            - baseline_f1
        )

        model_is_better = (
            accuracy_improvement > 0
            and f1_improvement > 0
        )

        return {
            "baseline": baseline_metrics,
            "model": model_metrics,
            "accuracy_improvement": (
                accuracy_improvement
            ),
            "f1_improvement": (
                f1_improvement
            ),
            "model_is_better": bool(
                model_is_better
            ),
            "quality": (
                "useful"
                if model_is_better
                else "needs_review"
            ),
        }

    raise ValueError(
        f"Unsupported prediction task: {task}"
    )


# =========================================================
# TRAINING
# =========================================================

def train_predictive_model(
    dataframe: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train the bounded predictive model.

    Large datasets are deterministically downsampled for the
    baseline model. The source DataFrame is never modified.
    """

    validate_dataframe(
        dataframe
    )

    readiness = can_run_prediction(
        dataframe
    )

    if not readiness["available"]:

        raise ValueError(
            readiness["reason"]
        )

    _validate_target(
        dataframe,
        target,
    )

    task = detect_prediction_task(
        dataframe,
        target,
    )

    # ---------------------------------------------------------
    # Bounded modeling dataset
    # ---------------------------------------------------------

    modeling_dataframe = dataframe

    was_sampled = False

    if len(
        modeling_dataframe
    ) > MAX_MODEL_ROWS:

        modeling_dataframe = (
            modeling_dataframe
            .sample(
                n=MAX_MODEL_ROWS,
                random_state=random_state,
            )
            .reset_index(
                drop=True
            )
        )

        was_sampled = True

    (
        X,
        y,
        numeric_features,
        categorical_features,
    ) = prepare_features(
        modeling_dataframe,
        target,
        max_categorical_levels=MAX_CATEGORICAL_LEVELS,
    )

    used_feature_columns = [
        *numeric_features,
        *categorical_features,
    ]

    excluded_features = [
        column
        for column in modeling_dataframe.columns
        if (
            column != target
            and column
            not in used_feature_columns
        )
    ]

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        X,
        y,
        task,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    model = build_model(
        task
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    if task == "regression":

        mae = float(
            mean_absolute_error(
                y_test,
                predictions,
            )
        )

        mse = float(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        rmse = mse ** 0.5

        try:
            r2 = float(
                r2_score(
                    y_test,
                    predictions,
                )
            )
        except Exception:
            r2 = float("nan")

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    else:

        metrics = {
            "accuracy": float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline_metrics = evaluate_baseline(
        y_train,
        y_test,
        task,
    )

    model_quality = compare_model_with_baseline(
        metrics,
        baseline_metrics,
        task,
    )

    # ---------------------------------------------------------
    # Safe output
    # ---------------------------------------------------------

    safe_predictions = [
        value.item()
        if hasattr(
            value,
            "item",
        )
        else value
        for value in predictions
    ]

    safe_actual_values = [
        value.item()
        if hasattr(
            value,
            "item",
        )
        else value
        for value in y_test
    ]

    return {
        "status": "completed",
        "task": task,
        "target": target,
        "model_name": type(
            model
        ).__name__,

        "feature_columns": used_feature_columns,

        "numeric_features": numeric_features,

        "categorical_features": categorical_features,

        "excluded_features": excluded_features,

        "train_rows": int(
            len(X_train)
        ),

        "test_rows": int(
            len(X_test)
        ),

        "source_rows": int(
            len(dataframe)
        ),

        "modeling_rows": int(
            len(modeling_dataframe)
        ),

        "was_sampled": was_sampled,

        "metrics": metrics,

        "baseline_metrics": baseline_metrics,

        "model_quality": model_quality,

        "predictions": safe_predictions,

        "actual_values": safe_actual_values,

        "model": pipeline,
    }


# =========================================================
# PUBLIC API
# =========================================================

def run_prediction(
    dataframe: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    """
    Run predictive analysis and return UI-safe results.
    """

    validate_dataframe(
        dataframe
    )

    _validate_target(
        dataframe,
        target,
    )

    result = train_predictive_model(
        dataframe,
        target,
    )

    # Only expose serializable/user-facing values.
    return {
        "status": result["status"],
        "task": result["task"],
        "target": result["target"],
        "model_name": result["model_name"],

        "feature_columns": result[
            "feature_columns"
        ],

        "numeric_features": result[
            "numeric_features"
        ],

        "categorical_features": result[
            "categorical_features"
        ],

        "excluded_features": result[
            "excluded_features"
        ],

        "train_rows": result[
            "train_rows"
        ],

        "test_rows": result[
            "test_rows"
        ],

        "source_rows": result[
            "source_rows"
        ],

        "modeling_rows": result[
            "modeling_rows"
        ],

        "was_sampled": result[
            "was_sampled"
        ],

        "metrics": result[
            "metrics"
        ],

        "baseline_metrics": result[
            "baseline_metrics"
        ],

        "model_quality": result[
            "model_quality"
        ],

        "predictions": result[
            "predictions"
        ],

        "actual_values": result[
            "actual_values"
        ],
    }