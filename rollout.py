"""Trajectory rollout utilities for the price negotiation environment.

This module provides helper functions and the main ``run_rollout`` entry point
for executing a complete buyer-seller negotiation episode against a live
environment server.

It is used by both ``inference.py`` (async, evaluation-oriented) and any
custom training loop that needs synchronous rollouts.

Rollout flow
------------
1. Connect to the server and call ``env.reset()`` to start a new episode.
2. On the first turn, send a canned opener via ``initial_buyer_message()``
   (avoids wasting an LLM call on a trivial greeting).
3. On subsequent turns, call ``get_openai_response()`` with the buyer's
   accumulated chat history to generate the next move.
4. After each ``env.step()``, record a ``TrajectoryStep`` snapshot.
5. Stop when ``step_result.done`` is ``True`` or the turn limit is reached.
6. Return a ``TrajectoryResult`` containing the full episode history.
"""

from __future__ import annotations

from typing import Literal

from price_negotiation import (
    PriceNegotiationAction,
    PriceNegotiationEnv,
    PriceNegotiationState,
)
from price_negotiation.server.helper_functions import get_openai_response
from price_negotiation.trajectory_types import TrajectoryResult, TrajectoryStep


# ---------------------------------------------------------------------------
# State inspection helpers
# ---------------------------------------------------------------------------

def format_product_name(state: PriceNegotiationState) -> str:
    """Extract a human-readable product name from the environment state.

    Looks up ``state.product_info["product"]["name"]`` and falls back to
    ``"Unknown Product"`` if either key is missing.

    Args:
        state: The current ``PriceNegotiationState`` returned by ``env.state()``.

    Returns:
        The product name string, or ``"Unknown Product"`` if not available.
    """
    return state.product_info.get("product", {}).get("name", "Unknown Product")


def latest_seller_reply(state: PriceNegotiationState) -> str | None:
    """Return the most recent seller message from the seller-side chat history.

    Scans ``state.seller_messages`` in reverse order and returns the content
    of the first ``assistant`` turn found (the seller LLM is the assistant in
    the seller-side history).

    Args:
        state: The current ``PriceNegotiationState`` returned by ``env.state()``.

    Returns:
        The seller's latest reply as a string, or ``None`` if the seller has
        not yet responded (e.g. immediately after ``reset()``).
    """
    for message in reversed(state.seller_messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return None


def initial_buyer_message(state: PriceNegotiationState) -> str | None:
    """Return a canned opening message for the very first buyer turn.

    Detects whether the buyer's chat history contains only the system prompt
    (i.e. no turns have been taken yet) and, if so, returns a generic
    interest-expression opener.  This avoids spending an LLM call on a
    trivial first message.

    On all subsequent turns (where the last message is not a ``system``
    message) this function returns ``None``, signalling to the caller that
    the LLM should generate the response instead.

    Args:
        state: The current ``PriceNegotiationState`` returned by ``env.state()``.

    Returns:
        A canned opener string on the very first turn, or ``None`` on all
        subsequent turns (or if ``buyer_messages`` is empty).
    """
    if not state.buyer_messages:
        return None

    last_message = state.buyer_messages[-1]
    if last_message.get("role") != "system":
        # At least one turn has already been taken; let the LLM respond.
        return None

    product_name = format_product_name(state)
    return (
        f"I am really interested in the {product_name} and would like "
        "to know more about this."
    )


# ---------------------------------------------------------------------------
# Main rollout entry point
# ---------------------------------------------------------------------------

def run_rollout(
    base_url: str,
    buyer_model: str,
    temperature: float,
    max_turns: int | None,
    difficulty: Literal["easy", "medium", "hard"] | None = None,
) -> TrajectoryResult:
    """Connect to the environment server and run one complete negotiation episode.

    Opens a synchronous WebSocket connection to the server, resets the
    environment (optionally filtering by difficulty), then loops until the
    episode terminates or the turn limit is reached.

    Turn generation strategy:
    - **Turn 1**: ``initial_buyer_message()`` returns a canned opener so no
      LLM call is needed.
    - **Turn 2+**: ``get_openai_response()`` is called with the full
      buyer-side chat history to generate the next move.

    Turn limit resolution (in priority order):
    1. The ``max_turns`` argument if explicitly provided (not ``None``).
    2. The ``metadata.max_turns`` field from the sampled dataset scenario.
    3. No limit (the loop runs until ``done=True``).

    Args:
        base_url: Base URL of the running environment server, e.g.
            ``"http://localhost:8000"`` or a Hugging Face Space URL.
        buyer_model: Model identifier passed to ``get_openai_response()`` for
            buyer-turn generation (e.g. ``"Qwen/Qwen2.5-72B-Instruct"``).
        temperature: Sampling temperature for the buyer LLM.  Lower values
            produce more deterministic offers; higher values increase variety.
        max_turns: Hard cap on the number of buyer turns.  Pass ``None`` to
            defer to the dataset scenario's ``metadata.max_turns`` value.
        difficulty: Optional difficulty filter for scenario sampling.  One of
            ``"easy"``, ``"medium"``, ``"hard"``, or ``None`` to sample from
            the full dataset.

    Returns:
        A ``TrajectoryResult`` containing the episode ID, the initial
        observation, the final state snapshot, and a ``TrajectoryStep`` for
        every buyer turn taken.
    """
    with PriceNegotiationEnv(base_url=base_url).sync() as env:
        reset_result = env.reset(difficulty=difficulty)
        state = env.state()
        steps: list[TrajectoryStep] = []

        # Prefer the caller-supplied limit; fall back to the dataset value.
        dataset_max_turns = state.product_info.get("metadata", {}).get("max_turns")
        turn_limit = max_turns if max_turns is not None else dataset_max_turns

        while True:
            state = env.state()

            # Enforce the turn limit before generating a response.
            if turn_limit is not None and state.step_count >= turn_limit:
                break

            # Use the canned opener on turn 1; call the LLM on all later turns.
            buyer_response = initial_buyer_message(state)
            if buyer_response is None:
                buyer_response = get_openai_response(
                    state.buyer_messages,
                    model=buyer_model,
                    temperature=temperature,
                )

            step_result = env.step(
                PriceNegotiationAction(buyer_response=buyer_response)
            )
            # Refresh state after the step so seller_reply reflects the new turn.
            state = env.state()

            steps.append(
                TrajectoryStep(
                    buyer_response=buyer_response,
                    observation=step_result.observation,
                    # Deep-copy so each step captures an independent snapshot.
                    state=state.model_copy(deep=True),
                    seller_reply=latest_seller_reply(state),
                )
            )

            if step_result.done:
                break

        return TrajectoryResult(
            episode_id=state.episode_id,
            initial_observation=reset_result.observation,
            final_state=state.model_copy(deep=True),
            steps=steps,
        )
