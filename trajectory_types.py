"""Shared trajectory result types for rollout and reward code."""

from __future__ import annotations

from dataclasses import dataclass

from price_negotiation import (
    PriceNegotiationObservation,
    PriceNegotiationState,
)


@dataclass
class TrajectoryStep:
    """One buyer action and the resulting observation/state."""

    buyer_response: str
    observation: PriceNegotiationObservation
    state: PriceNegotiationState
    seller_reply: str | None = None


@dataclass
class TrajectoryResult:
    """Full rollout result for one negotiation episode."""

    episode_id: str | None
    initial_observation: PriceNegotiationObservation
    final_state: PriceNegotiationState
    steps: list[TrajectoryStep]
