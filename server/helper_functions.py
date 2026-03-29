"""Helper utilities for model-backed negotiation behavior."""

import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def get_openai_response(
    messages: list[dict[str, str]],
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.7,
) -> str:
    """Send a chat history to OpenAI and return the assistant response text."""
    if OpenAI is None:
        raise ImportError(
            "openai is not installed. Add the dependency and install the project again."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response: Any = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI response did not contain any message content.")

    return content
