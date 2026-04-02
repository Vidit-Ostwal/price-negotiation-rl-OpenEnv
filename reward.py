"""Trajectory-level reward helpers for price negotiation."""

from __future__ import annotations

import re
from typing import Optional

try:
    from price_negotiation.trajectory_types import TrajectoryResult
except ImportError:
    from trajectory_types import TrajectoryResult


MAX_TURNS_DEFAULT = 10
Messages = list[dict[str, str]]

PRICE_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")
ACTION_RE = re.compile(
    r"<action>(OFFER\s*\$?([\d,]+(?:\.\d+)?)|ACCEPT|WALK)</action>",
    re.IGNORECASE,
)


def extract_last_price(text: str | None) -> float:
    """Extract the last dollar price mentioned in a message."""
    if not text:
        return 0.0

    matches = PRICE_PATTERN.findall(text)
    if not matches:
        return 0.0

    return float(matches[-1])


def _infer_final_price_optional(trajectory: TrajectoryResult) -> float | None:
    """Infer the accepted final price from buyer/seller offer histories."""
    if not trajectory.steps:
        return None

    terminal_step = trajectory.steps[-1]
    if terminal_step.observation.deal_status != "ACCEPTED":
        return None

    buyer_history = trajectory.final_state.buyer_messages
    seller_history = trajectory.final_state.seller_messages
    latest_buyer_offer = _get_latest_offer_from_messages(
        buyer_history, role="assistant"
    )
    latest_seller_offer = _get_latest_offer_from_messages(
        seller_history, role="assistant"
    )

    buyer_action, _ = _parse_action(terminal_step.buyer_response)
    if buyer_action == "ACCEPT":
        return latest_seller_offer

    seller_action, _ = _parse_action(terminal_step.seller_reply or "")
    if seller_action == "ACCEPT":
        return latest_buyer_offer

    return latest_seller_offer or latest_buyer_offer


def infer_final_price(trajectory: TrajectoryResult) -> float:
    """Return a numeric final price for reward consumers."""
    return _infer_final_price_optional(trajectory) or 0.0


def buyer_completion_messages(trajectory: TrajectoryResult) -> Messages:
    """Return buyer-side chat history for reward evaluation."""
    return trajectory.final_state.buyer_messages


def reward_state(trajectory: TrajectoryResult) -> dict:
    """Build episode-level state expected by reward functions."""
    product_info = trajectory.final_state.product_info
    valuations = product_info.get("valuations", {})
    metadata = product_info.get("metadata", {})
    final_observation = trajectory.steps[-1].observation if trajectory.steps else None

    return {
        "deal_reached": (
            final_observation is not None
            and final_observation.deal_status == "ACCEPTED"
        ),
        "deal_possible": valuations.get("deal_possible"),
        "buyer_true_value": valuations.get("buyer_true_value"),
        "final_price": infer_final_price(trajectory),
        "final_price_valid": _infer_final_price_optional(trajectory) is not None,
        "zopa_width": valuations.get("zopa_width", 0),
        "turn": final_observation.negotiation_round if final_observation else 0,
        "max_turns": metadata.get("max_turns", MAX_TURNS_DEFAULT),
    }


def _parse_action(message: str) -> tuple[str, Optional[float]]:
    match = ACTION_RE.search(message)
    if not match:
        return ("INVALID", None)

    raw = match.group(1).upper().strip()
    if raw == "ACCEPT":
        return ("ACCEPT", None)
    if raw == "WALK":
        return ("WALK", None)
    if raw.startswith("OFFER"):
        price_str = match.group(2)
        if price_str:
            return ("OFFER", float(price_str.replace(",", "")))
    return ("INVALID", None)


def _get_buyer_offers(completion: Messages) -> list[float]:
    """Extract ordered list of prices from buyer OFFER turns."""
    offers = []
    for message in completion:
        if message["role"] == "assistant":
            action, price = _parse_action(message["content"])
            if action == "OFFER" and price is not None:
                offers.append(price)
    return offers


def _get_latest_offer_from_messages(
    messages: Messages,
    role: str = "assistant",
) -> float | None:
    """Extract the latest offer price from one side of the conversation."""
    for message in reversed(messages):
        if message["role"] != role:
            continue
        action, price = _parse_action(message["content"])
        if action == "OFFER" and price is not None:
            return price
    return None


def surplus_reward(completion: Messages, info: dict, **kwargs) -> float:
    """Reward buyer surplus on successful deals."""
    state = kwargs.get("state", {})
    if not state.get("deal_reached"):
        return 0.0
    if not state.get("final_price_valid", False):
        return 0.0
    if state["final_price"] > state["buyer_true_value"]:
        return -1.0
    zopa_width = state["zopa_width"]
    if zopa_width <= 0:
        return 0.0
    surplus = (state["buyer_true_value"] - state["final_price"]) / zopa_width
    return float(max(-1.0, min(1.0, surplus)))


def walkaway_penalty(completion: Messages, info: dict, **kwargs) -> float:
    """Reward or penalize the final deal/walk outcome."""
    state = kwargs.get("state", {})
    deal_reached = bool(state.get("deal_reached", False))
    deal_possible = bool(state.get("deal_possible", True))

    if deal_possible and deal_reached:
        return 1.0
    if deal_possible and not deal_reached:
        return -5.0
    if not deal_possible and not deal_reached:
        return 1.0
    if not deal_possible and deal_reached:
        return 5.0
    return 0.0


def format_reward(completion: Messages, info: dict, **kwargs) -> float:
    """Reward compliance with required buyer action-tag formatting."""
    buyer_turns = [message for message in completion if message["role"] == "assistant"]
    if not buyer_turns:
        return 0.0
    valid = sum(
        1 for message in buyer_turns if _parse_action(message["content"])[0] != "INVALID"
    )
    return valid / len(buyer_turns)


def efficiency_bonus(completion: Messages, info: dict, **kwargs) -> float:
    """Reward closing a deal in fewer turns."""
    state = kwargs.get("state", {})
    if not state.get("deal_reached"):
        return 0.0
    max_turns = (
        state.get("max_turns")
        or info.get("metadata", {}).get("max_turns")
        or MAX_TURNS_DEFAULT
    )
    return (max_turns - state.get("turn", max_turns)) / max_turns


def anchoring_reward(completion: Messages, info: dict, **kwargs) -> float:
    """Reward opening with a strategically low anchor."""
    state = kwargs.get("state", {})
    buyer_true_value = (
        state.get("buyer_true_value")
        or info.get("valuations", {}).get("buyer_true_value")
    )
    if not buyer_true_value:
        return 0.0

    buyer_offers = _get_buyer_offers(completion)
    if not buyer_offers:
        return 0.0

    opening_offer = buyer_offers[0]
    ideal = 0.65 * buyer_true_value
    distance = abs(opening_offer - ideal) / buyer_true_value
    return float(1.0 - 2.0 * distance)


def negotiation_progress_reward(
    completion: Messages, info: dict, **kwargs
) -> float:
    """Reward controlled upward concessions and penalize backtracking."""
    state = kwargs.get("state", {})
    buyer_true_value = state.get("buyer_true_value")
    if not buyer_true_value:
        return 0.0

    buyer_offers = _get_buyer_offers(completion)
    if len(buyer_offers) < 2:
        return 0.0

    alpha = 4.0
    rewards = []

    for index in range(1, len(buyer_offers)):
        prev_offer = buyer_offers[index - 1]
        curr_offer = buyer_offers[index]
        delta = curr_offer - prev_offer

        if delta < 0:
            rewards.append(-1.0)
            continue

        ratio = delta / buyer_true_value
        step_score = 1.0 - alpha * ratio
        rewards.append(step_score)

    return float(sum(rewards) / len(rewards))


def reward_breakdown(trajectory: TrajectoryResult) -> dict[str, float]:
    """Compute all trajectory-level reward components."""
    completion = buyer_completion_messages(trajectory)
    info = trajectory.final_state.product_info
    state = reward_state(trajectory)

    return {
        "surplus_reward": surplus_reward(completion, info, state=state),
        "walkaway_penalty": walkaway_penalty(completion, info, state=state),
        "format_reward": format_reward(completion, info, state=state),
        "efficiency_bonus": efficiency_bonus(completion, info, state=state),
        "anchoring_reward": anchoring_reward(completion, info, state=state),
        "negotiation_progress_reward": negotiation_progress_reward(
            completion, info, state=state
        ),
    }


def score_trajectory(trajectory: TrajectoryResult) -> float:
    """Aggregate the full trajectory reward from component scores."""
    breakdown = reward_breakdown(trajectory)
    return float(sum(breakdown.values()))
