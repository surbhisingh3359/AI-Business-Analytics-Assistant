from src.preprocessing import preprocess_data
from src.ai_context import build_ai_context
from src.prompt_builder import build_business_prompt
from src.llm_client import LLMClient
from src.response_validator import validate_ai_response


def analyze_business_data(file_path: str) -> dict:
    """Run the complete AI-powered business analysis pipeline."""

    preprocessing_result = preprocess_data(file_path)

    df = preprocessing_result["data"]

    context = build_ai_context(df)

    prompt = build_business_prompt(context)

    client = LLMClient()

    response = client.generate(prompt)

    validation = validate_ai_response(
        response,
        context,
    )

    return {
        "data": df,
        "quality_report": preprocessing_result["quality_report"],
        "context": context,
        "prompt": prompt,
        "analysis": response,
        "validation": validation,
    }