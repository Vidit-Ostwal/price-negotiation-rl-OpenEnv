"""Shared trajectory result types for rollout and reward code.

This module defines the two dataclasses used to represent a completed
negotiation episode:

- ``TrajectoryStep``   — a single buyer turn with its resulting observation
                         and state snapshot
- ``TrajectoryResult`` — the full episode, composed of all steps plus the
                         initial observation and final state

These types are intentionally kept as plain Python dataclasses (rather than
Pydantic models) so they are lightweight and have no dependency on the
OpenEnv runtime.  They are produced by ``rollout.py`` / ``inference.py`` and
consumed by ``reward.py``.

Typical data flow
-----------------
::

    rollout.run_rollout(...)          # produces TrajectoryResult
        └─ TrajectoryStep × N         # one per buyer turn

    reward.reward_breakdown(result)   # consumes TrajectoryResult
    reward.score_trajectory(result)   # consumes TrajectoryResult
"""

from __future__ import annotations

from dataclasses import dataclass

from price_negotiation import (
    PriceNegotiationObservation,
    PriceNegotiationState,
)


@dataclass
class TrajectoryStep:
    """A single buyer turn and its immediate outcome.

    Captures everything that happened during one call to ``env.step()``:
    the buyer's action, the observation returned by the server, a deep-copy
    of the environment state *after* the step (including the seller's reply
    appended to the chat histories), and the seller's reply extracted as a
    convenience string.

    Attributes:
        buyer_response: The raw text sent by the buyer agent, including any
            action tag (``<action>OFFER $X</action>``, ``<action>ACCEPT</action>``,
            or ``<action>WALK</action>``).
        observation: The ``PriceNegotiationObservation`` returned by
            ``env.step()``.  Contains ``deal_status``, ``negotiation_round``,
            ``next_turn``, ``done``, and ``reward``.
        state: A deep-copy of the ``PriceNegotiationState`` *after* this step
            was processed.  Includes the updated ``buyer_messages`` and
            ``seller_messages`` (so the seller's reply is already appended).
        seller_reply: The seller's latest assistant message extracted from
            ``state.seller_messages`` for convenience.  ``None`` if the
            episode terminated before the seller responded (e.g. the buyer
            walked away or accepted).
    """

    buyer_response: str
    observation: PriceNegotiationObservation
    state: PriceNegotiationState
    seller_reply: str | None = None


@dataclass
class TrajectoryResult:
    """The complete record of one negotiation episode.

    Produced by ``rollout.run_rollout()`` and ``inference.py`` after an
    episode finishes.  Contains everything needed to compute reward scores
    offline without re-running the environment.

    Attributes:
        episode_id: The UUID assigned to this episode by the server, or
            ``None`` if the server did not return one.  Useful for
            reproducibility and logging.
        initial_observation: The ``PriceNegotiationObservation`` returned by
            ``env.reset()``.  Always has ``deal_status="ONGOING"``,
            ``negotiation_round=0``, and ``done=False``.
        final_state: A deep-copy of the ``PriceNegotiationState`` at the end
            of the episode.  Contains the complete ``buyer_messages`` and
            ``seller_messages`` histories, the full ``product_info`` (including
            private valuations), and the final ``step_count``.
        steps: Ordered list of ``TrajectoryStep`` objects, one per buyer turn.
            ``steps[-1].observation.done`` is always ``True`` for a normally
            terminated episode (it may be ``False`` if the rollout was stopped
            by an external turn limit).

    Example::

        from price_negotiation.reward import reward_breakdown, score_trajectory

        result: TrajectoryResult = run_rollout(...)
        breakdown = reward_breakdown(result)   # dict of component scores
        score     = score_trajectory(result)   # single float in [0, 1]
    """

    episode_id: str | None
    initial_observation: PriceNegotiationObservation
    final_state: PriceNegotiationState
    steps: list[TrajectoryStep]
