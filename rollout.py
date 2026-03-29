"""Trajectory rollout utilities for the price negotiation environment."""

from __future__ import annotations

from price_negotiation import (
    PriceNegotiationAction,
    PriceNegotiationEnv,
    PriceNegotiationState,
)
from price_negotiation.server.helper_functions import get_openai_response
from price_negotiation.trajectory_types import TrajectoryResult, TrajectoryStep


def format_product_name(state: PriceNegotiationState) -> str:
    """Extract a readable product name from environment state."""
    return state.product_info.get("product", {}).get("name", "Unknown Product")


def latest_seller_reply(state: PriceNegotiationState) -> str | None:
    """Return the latest seller assistant message, if present."""
    for message in reversed(state.seller_messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return None


def initial_buyer_message(state: PriceNegotiationState) -> str | None:
    """Return the buyer opening message when only the system prompt is present."""
    if not state.buyer_messages:
        return None

    last_message = state.buyer_messages[-1]
    if last_message.get("role") != "system":
        return None

    product_name = format_product_name(state)
    return (
        f"I am really interested in the {product_name} and would like "
        "to know more about this."
    )


def run_rollout(
    base_url: str,
    buyer_model: str,
    temperature: float,
    max_turns: int | None,
) -> TrajectoryResult:
    """Connect to the server and run one full buyer-seller trajectory."""
    with PriceNegotiationEnv(base_url=base_url).sync() as env:
        reset_result = env.reset()
        state = env.state()
        steps: list[TrajectoryStep] = []

        dataset_max_turns = state.product_info.get("metadata", {}).get("max_turns")
        turn_limit = max_turns if max_turns is not None else dataset_max_turns

        while True:
            state = env.state()

            if turn_limit is not None and state.step_count >= turn_limit:
                break

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
            state = env.state()

            steps.append(
                TrajectoryStep(
                    buyer_response=buyer_response,
                    observation=step_result.observation,
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
