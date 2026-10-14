import os

import ollama


class LLMClient:
    """Client for generating business analysis using a local Ollama model."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    def generate(self, prompt: str) -> str:
        """Generate an AI response using the local Ollama model."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        if not prompt.strip():
            raise ValueError("prompt cannot be empty.")

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]
