"""Helper utilities for model-backed negotiation behavior.

This module provides a thin wrapper around the OpenAI-compatible chat
completions API used by both the seller (server-side) and the buyer
(client/inference-side) during a negotiation episode.

Environment variables consumed here:
    API_KEY      – Primary API key for the inference endpoint.
                   Falls back to HF_TOKEN when API_KEY is not set.
    HF_TOKEN     – Hugging Face token used as a fallback API key.
    API_BASE_URL – Base URL of the OpenAI-compatible endpoint.
                   Defaults to the Hugging Face inference router
                   (https://router.huggingface.co/v1).
    SELLER_MODEL – Model identifier used for seller-side generation.
                   Defaults to DEFAULT_OPENAI_MODEL when not set.
"""

import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    # openai is an optional dependency at import time; callers that actually
    # invoke get_openai_response will receive a clear ImportError at runtime.
    OpenAI = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

# Default model used for both buyer and seller when no override is provided.
DEFAULT_OPENAI_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# API key resolved from the environment.  API_KEY takes precedence; HF_TOKEN
# is accepted as a convenience alias for Hugging Face-hosted deployments.
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

# Base URL of the OpenAI-compatible inference endpoint.  Override this to
# point at a local vLLM server, Together AI, or any other compatible host.
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"

# Model used specifically for seller-side generation inside the environment.
# Can be overridden independently of the buyer model.
SELLER_MODEL = os.getenv("SELLER_MODEL", DEFAULT_OPENAI_MODEL)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_openai_response(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Send a chat history to an OpenAI-compatible endpoint and return the reply.

    Constructs a fresh ``OpenAI`` client on every call using the module-level
    ``API_KEY`` and ``API_BASE_URL`` values so that runtime environment changes
    are always picked up without restarting the process.

    Args:
        messages: Ordered list of chat messages in OpenAI format, e.g.
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``.
            The list is passed directly to ``chat.completions.create`` and must
            contain at least one message.
        model: Model identifier to use for this request.  When ``None`` the
            module-level ``DEFAULT_OPENAI_MODEL`` is used.
        temperature: Sampling temperature forwarded to the model.  Lower values
            produce more deterministic output; higher values increase diversity.
            Defaults to ``0.7``.

    Returns:
        The assistant's reply as a plain string (leading/trailing whitespace
        preserved as returned by the model).

    Raises:
        ImportError: If the ``openai`` package is not installed.
        ValueError: If ``API_KEY`` / ``HF_TOKEN`` is not set in the environment,
            if the API response contains no choices, if the message content is
            ``None``, or if the content is an empty/whitespace-only string.
    """
    if OpenAI is None:
        raise ImportError(
            "openai is not installed. Add the dependency and install the project again."
        )

    if not API_KEY:
        raise ValueError("API_KEY or HF_TOKEN is not set.")

    # Fall back to the default model when the caller does not specify one.
    resolved_model = model or DEFAULT_OPENAI_MODEL

    # Build client kwargs; base_url is optional so only include it when set.
    client_kwargs: dict[str, str] = {"api_key": API_KEY}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL

    client = OpenAI(**client_kwargs)
    response: Any = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
    )

    # Guard against malformed or empty API responses before accessing fields.
    if not getattr(response, "choices", None):
        raise ValueError("OpenAI response did not contain any choices.")

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI response did not contain any message content.")
    if not content.strip():
        raise ValueError("OpenAI response content was empty.")

    return content


def check_openai_response(model: str | None = None) -> bool:
    """Probe the configured inference endpoint to verify connectivity.

    Sends a minimal single-turn request ("Reply with OK only.") and returns
    ``True`` when a non-empty response is received.  Intended for use in
    health-check scripts or startup validation, not in the hot path.

    Args:
        model: Optional model identifier to test.  Falls back to
            ``DEFAULT_OPENAI_MODEL`` when ``None``.

    Returns:
        ``True`` if the endpoint returned a non-empty response, ``False`` if
        any exception was raised (connection error, auth failure, etc.).
    """
    try:
        response = get_openai_response(
            messages=[{"role": "user", "content": "Reply with OK only."}],
            model=model,
            temperature=0.0,
        )
    except Exception as Error:
        print(Error)
        return False

    return bool(response.strip())
