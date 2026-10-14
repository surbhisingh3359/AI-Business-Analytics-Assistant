import io
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder

from src.business_analysis_pipeline import run_business_analysis
from src.ai_analysis import analyze_business_data


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
PREVIEW_ROW_LIMIT = 100


def _load_uploaded_dataframe(
    contents: bytes,
    filename: str,
) -> pd.DataFrame:
    """Read one supported upload without writing it to disk."""

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only CSV and XLSX files are supported.",
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        buffer = io.BytesIO(contents)
        dataframe = (
            pd.read_csv(buffer)
            if suffix == ".csv"
            else pd.read_excel(buffer)
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded file could not be read: {exc}",
        ) from exc

    if dataframe.empty:
        raise HTTPException(
            status_code=400,
            detail="The dataset contains no usable rows.",
        )

    return dataframe


def _analysis_response(
    dataframe: pd.DataFrame,
    filename: str,
) -> dict[str, Any]:
    """Build a bounded, JSON-safe API response from the canonical pipeline."""

    result = dict(run_business_analysis(dataframe))
    result.pop("data", None)

    return jsonable_encoder(
        {
            "filename": filename,
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "column_names": [str(column) for column in dataframe.columns],
            "data_preview": dataframe.head(PREVIEW_ROW_LIMIT).to_dict(
                orient="records"
            ),
            "analysis": result,
        }
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AI Business Analytics Assistant API",
    description=(
        "API for deterministic, evidence-led business analysis. "
        "Optional LLM narration remains available through /ai-analysis."
    ),
    version="1.0.0",
)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "name": "AI Business Analytics Assistant API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Check whether the API is running."""

    return {
        "status": "healthy",
    }


# =========================================================
# DATASET SUMMARY
# =========================================================

@app.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Upload a CSV or XLSX file and return the canonical business analysis.

    This endpoint does not call the LLM.
    It is intended for fast, deterministic dataset analysis.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    try:
        contents = await file.read()
        dataframe = _load_uploaded_dataframe(contents, file.filename)
        return _analysis_response(dataframe, file.filename)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset analysis failed: {exc}",
        ) from exc

# =========================================================
# AI BUSINESS ANALYSIS
# =========================================================

@app.post("/ai-analysis", deprecated=True)
async def ai_analysis(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Upload a CSV file and generate the complete
    AI-powered business analysis.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    temp_path: str | None = None

    try:
        with NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        result = analyze_business_data(
            temp_path
        )

        # DataFrames are not directly JSON serializable.
        data = result.get("data")

        if isinstance(data, pd.DataFrame):
            result["data"] = data.to_dict(
                orient="records"
            )

        return {
            "filename": file.filename,
            **result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {exc}",
        ) from exc

    finally:
        if temp_path is not None:
            path = Path(temp_path)

            if path.exists():
                path.unlink()
