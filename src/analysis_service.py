from typing import Any

from src.preprocessing import preprocess_data
from src.ai_context import build_ai_context
from src.prompt_builder import build_business_prompt
from src.llm_client import LLMClient
from src.response_validator import validate_ai_response


def run_analysis(
    file_path: str,
    model: str = "qwen2.5:3b",
) -> dict[str, Any]:
    """
    Run the complete AI Business Analytics pipeline.

    Pipeline:
        File
        -> preprocessing
        -> AI context
        -> business prompt
        -> LLM
        -> response validation
        -> structured result
    """

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")

    if not file_path.strip():
        raise ValueError("file_path cannot be empty.")

    if not isinstance(model, str):
        raise TypeError("model must be a string.")

    if not model.strip():
        raise ValueError("model cannot be empty.")

    # ---------------------------------------------------------
    # 1. Load and preprocess dataset
    # ---------------------------------------------------------

    preprocessing_result = preprocess_data(
        file_path
    )

    if not isinstance(
        preprocessing_result,
        dict,
    ):
        raise TypeError(
            "preprocess_data must return a dictionary."
        )

    if "data" not in preprocessing_result:
        raise ValueError(
            "Preprocessing result is missing data."
        )

    df = preprocessing_result["data"]

    # ---------------------------------------------------------
    # 2. Build complete AI context
    # ---------------------------------------------------------

    context = build_ai_context(df)

    if not isinstance(context, dict):
        raise TypeError(
            "build_ai_context must return a dictionary."
        )

    # ---------------------------------------------------------
    # 3. Build grounded business prompt
    # ---------------------------------------------------------

    prompt = build_business_prompt(
        context
    )

    if not isinstance(prompt, str):
        raise TypeError(
            "build_business_prompt must return a string."
        )

    # ---------------------------------------------------------
    # 4. Generate AI analysis
    # ---------------------------------------------------------

    client = LLMClient(
        model=model
    )

    response = client.generate(
        prompt
    )

    # ---------------------------------------------------------
    # 5. Validate AI response
    # ---------------------------------------------------------

    validation = validate_ai_response(
        response,
        context,
    )

    # ---------------------------------------------------------
    # 6. Return structured result
    # ---------------------------------------------------------

    return {
        "data": df,
        "quality_report": preprocessing_result.get(
            "quality_report",
            {},
        ),
        "context": context,
        "prompt": prompt,
        "analysis": response,
        "validation": validation,
    }