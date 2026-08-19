import pytest

from src.llm_client import LLMClient


def test_llm_client_initialization():
    client = LLMClient()

    assert client.model == "qwen2.5:3b"


def test_llm_client_custom_model():
    client = LLMClient(model="custom-model")

    assert client.model == "custom-model"


def test_llm_client_rejects_invalid_prompt():
    client = LLMClient()

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        client.generate("")


def test_llm_client_rejects_non_string_prompt():
    client = LLMClient()

    with pytest.raises(TypeError, match="prompt must be a string"):
        client.generate(None)