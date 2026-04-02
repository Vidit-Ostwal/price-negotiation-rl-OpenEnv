"""Helper utilities for model-backed negotiation behavior."""

import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


DEFAULT_OPENAI_MODEL = "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL")
SELLER_MODEL = os.getenv("SELLER_MODEL", DEFAULT_OPENAI_MODEL)


def get_openai_response(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Send a chat history to OpenAI and return the assistant response text."""
    if OpenAI is None:
        raise ImportError(
            "openai is not installed. Add the dependency and install the project again."
        )

    if not API_KEY:
        raise ValueError("API_KEY or HF_TOKEN is not set.")

    resolved_model = model or DEFAULT_OPENAI_MODEL

    client_kwargs: dict[str, str] = {"api_key": API_KEY}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL

    client = OpenAI(**client_kwargs)
    response: Any = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
    )

    if not getattr(response, "choices", None):
        raise ValueError("OpenAI response did not contain any choices.")

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI response did not contain any message content.")
    if not content.strip():
        raise ValueError("OpenAI response content was empty.")

    return content


def check_openai_response(model: str | None = None) -> bool:
    """Run a tiny probe request to verify the configured client path works."""
    try:
        response = get_openai_response(
            messages=[{"role": "user", "content": "Reply with OK only."}],
            model=model,
            temperature=0.0,
        )
    except Exception:
        return False

    return bool(response.strip())

print(check_openai_response())
