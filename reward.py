"""Trajectory-level reward helpers for price negotiation.

This module provides all reward computation logic for evaluating a completed
negotiation trajectory.  It is intentionally decoupled from the environment
server so it can be used in offline evaluation, RL training pipelines, or
any other context that has access to a ``TrajectoryResult``.

Public API
----------
- ``reward_breakdown(trajectory)``  — compute all six component scores
- ``score_trajectory(trajectory)``  — aggregate components into a single [0, 1] scalar
- ``reward_state(trajectory)``      — extract the episode-level dict used by reward fns
- ``infer_final_price(trajectory)`` — determine the agreed price (or 0.0 if no deal)
- ``extract_last_price(text)``      — parse the last ``$X`` mention from a string

Reward components
-----------------
Each component function has the signature
``(completion: Messages, info: dict, **kwargs) -> float`` where ``kwargs``
always contains a ``"state"`` key produced by ``reward_state()``.

+-------------------------------+------------------+------------------------------------------+
| Function                      | Raw range        | What it measures                         |
+===============================+==================+==========================================+
| ``surplus_reward``            | [-1, 1]          | Buyer surplus as fraction of ZOPA width  |
| ``walkaway_penalty``          | {-5, 1, 5}       | Correctness of the deal/walk decision    |
| ``format_reward``             | [0, 1]           | Fraction of turns with valid action tags |
| ``efficiency_bonus``          | [0, 1]           | Speed of closing (fewer turns = higher)  |
| ``anchoring_reward``          | [-1, 1]          | Quality of the opening anchor offer      |
| ``negotiation_progress_reward``| [-1, 1]         | Controlled upward concessions            |
+-------------------------------+------------------+------------------------------------------+

``score_trajectory`` normalises each component to [0, 1] and returns their
unweighted average.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    from price_negotiation.trajectory_types import TrajectoryResult
except ImportError:
    from trajectory_types import TrajectoryResult


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Fallback turn budget used when the dataset scenario does not specify max_turns.
MAX_TURNS_DEFAULT = 10

# Convenience type alias for an OpenAI-format chat history.
Messages = list[dict[str, str]]

# Matches a dollar amount anywhere in a string, e.g. "$1,250" or "$450.00".
PRICE_PATTERN = re.compile(r"\$(\d+(?:\.\d+)?)")

# Matches the three valid action tags produced by buyer/seller LLMs:
#   <action>OFFER $X</action>  — price offer (dollar sign optional in tag)
#   <action>ACCEPT</action>    — accept the current offer
#   <action>WALK</action>      — walk away
ACTION_RE = re.compile(
    r"<action>(OFFER\s*\$?([\d,]+(?:\.\d+)?)|ACCEPT|WALK)</action>",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the closed interval ``[low, high]``.

    Args:
        value: The scalar to clamp.
        low:   Lower bound (inclusive).
        high:  Upper bound (inclusive).

    Returns:
        ``value`` if it is already within ``[low, high]``, otherwise the
        nearest boundary.
    """
    return max(low, min(high, value))


def _parse_action(message: str) -> tuple[str, Optional[float]]:
    """Extract the action type and optional price from a message string.

    Searches for the first ``<action>...</action>`` tag in ``message`` and
    returns a ``(action_type, price)`` tuple.

    Args:
        message: Raw text from a buyer or seller turn, potentially containing
            an action tag.

    Returns:
        A tuple ``(action_type, price)`` where:
        - ``action_type`` is one of ``"OFFER"``, ``"ACCEPT"``, ``"WALK"``,
          or ``"INVALID"`` if no valid tag was found.
        - ``price`` is the parsed float for ``OFFER`` actions, or ``None``
          for ``ACCEPT``, ``WALK``, and ``INVALID``.
    """
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
            # Remove commas from numbers like "1,250" before converting.
            return ("OFFER", float(price_str.replace(",", "")))
    return ("INVALID", None)


def _get_buyer_offers(completion: Messages) -> list[float]:
    """Return an ordered list of all prices the buyer offered during the episode.

    Iterates through the buyer-side chat history in chronological order and
    collects the price from every ``OFFER`` action tag found in ``assistant``
    turns.

    Args:
        completion: Buyer-side chat history (``buyer_messages`` from state).

    Returns:
        A list of offer prices in the order they were made.  Empty if the
        buyer never made an explicit offer.
    """
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
    """Return the most recent offer price made by one side of the conversation.

    Scans the message list in reverse so the most recent matching turn is
    found first.

    Args:
        messages: Chat history for one side (buyer or seller).
        role:     The role whose turns to inspect.  Defaults to
                  ``"assistant"`` (the LLM-generated side of each history).

    Returns:
        The price from the most recent ``OFFER`` action tag, or ``None`` if
        no offer was found.
    """
    for message in reversed(messages):
        if message["role"] != role:
            continue
        action, price = _parse_action(message["content"])
        if action == "OFFER" and price is not None:
            return price
    return None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def extract_last_price(text: str | None) -> float:
    """Extract the last dollar price mentioned anywhere in ``text``.

    Uses a simple regex (``$X`` or ``$X.XX``) rather than action-tag parsing,
    so it works on free-form text as well as structured messages.

    Args:
        text: Any string that may contain dollar amounts, or ``None``.

    Returns:
        The last dollar amount found as a float, or ``0.0`` if ``text`` is
        ``None``, empty, or contains no dollar amounts.
    """
    if not text:
        return 0.0

    matches = PRICE_PATTERN.findall(text)
    if not matches:
        return 0.0

    return float(matches[-1])


def _infer_final_price_optional(trajectory: TrajectoryResult) -> float | None:
    """Infer the agreed price from the terminal step of a trajectory.

    Logic:
    - If the episode did not end with ``ACCEPTED``, returns ``None``.
    - If the buyer's terminal action was ``ACCEPT``, the agreed price is the
      seller's most recent offer.
    - If the seller's terminal action was ``ACCEPT``, the agreed price is the
      buyer's most recent offer.
    - As a fallback, returns whichever side's latest offer is available.

    Args:
        trajectory: The completed negotiation trajectory.

    Returns:
        The inferred final price as a float, or ``None`` if the deal was not
        accepted or no offer prices could be found.
    """
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
        # Buyer accepted → the price on the table was the seller's last offer.
        return latest_seller_offer

    seller_action, _ = _parse_action(terminal_step.seller_reply or "")
    if seller_action == "ACCEPT":
        # Seller accepted → the price on the table was the buyer's last offer.
        return latest_buyer_offer

    # Fallback: return whichever offer is available.
    return latest_seller_offer or latest_buyer_offer


def infer_final_price(trajectory: TrajectoryResult) -> float:
    """Return the agreed final price, or ``0.0`` if no deal was reached.

    Thin wrapper around ``_infer_final_price_optional`` that guarantees a
    numeric return value for use in reward calculations.

    Args:
        trajectory: The completed negotiation trajectory.

    Returns:
        The inferred final price as a float, or ``0.0`` if the deal was not
        accepted or no price could be inferred.
    """
    return _infer_final_price_optional(trajectory) or 0.0


def buyer_completion_messages(trajectory: TrajectoryResult) -> Messages:
    """Return the buyer-side chat history from a completed trajectory.

    Convenience accessor used by all reward functions to obtain the
    ``completion`` argument from a ``TrajectoryResult``.

    Args:
        trajectory: The completed negotiation trajectory.

    Returns:
        The buyer-side ``Messages`` list from ``trajectory.final_state``.
    """
    return trajectory.final_state.buyer_messages


def reward_state(trajectory: TrajectoryResult) -> dict:
    """Build the episode-level state dict consumed by all reward functions.

    Extracts and normalises the fields that reward functions need from the
    trajectory's final state and product info, providing safe defaults for
    any missing keys.

    Args:
        trajectory: The completed negotiation trajectory.

    Returns:
        A dict with the following keys:

        - ``deal_reached`` (bool): Whether the episode ended with ``ACCEPTED``.
        - ``deal_possible`` (bool | None): Whether a ZOPA existed (from valuations).
        - ``buyer_true_value`` (float | None): Buyer's private maximum willingness to pay.
        - ``final_price`` (float): Agreed price, or ``0.0`` if no deal.
        - ``final_price_valid`` (bool): ``True`` if a price could be inferred.
        - ``zopa_width`` (float): Width of the Zone of Possible Agreement.
        - ``turn`` (int): The negotiation round at episode end.
        - ``max_turns`` (int): Turn budget from metadata, or ``MAX_TURNS_DEFAULT``.
    """
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


# ---------------------------------------------------------------------------
# Reward component functions
# ---------------------------------------------------------------------------

def surplus_reward(completion: Messages, info: dict, **kwargs) -> float:
    """Score the buyer's economic surplus on a successful deal.

    Measures how much of the ZOPA the buyer captured.  A score of ``1.0``
    means the buyer paid the seller's reserve price (maximum surplus);
    ``0.0`` means the buyer paid their own true value (zero surplus);
    ``-1.0`` means the buyer paid above their true value.

    Returns ``0.0`` immediately if no deal was reached or if the final price
    could not be inferred.

    Args:
        completion: Buyer-side chat history (unused directly; provided for
            API consistency with other reward functions).
        info: Product info dict from the episode state (unused directly).
        **kwargs: Must contain ``state`` (dict from ``reward_state()``).

    Returns:
        A float in ``[-1.0, 1.0]``.
    """
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
    """Score the correctness of the final deal-or-walk decision.

    Rewards the agent for making the economically rational choice:
    - ``+1.0``  deal reached when a ZOPA existed (correct deal)
    - ``-5.0``  walked away when a ZOPA existed (missed deal)
    - ``+1.0``  walked away when no ZOPA existed (correct walk)
    - ``+5.0``  deal reached when no ZOPA existed (buyer overpaid — rare bonus
                because the seller accepted below reserve, which is unusual)

    Args:
        completion: Buyer-side chat history (unused directly).
        info: Product info dict (unused directly).
        **kwargs: Must contain ``state`` (dict from ``reward_state()``).

    Returns:
        One of ``{-5.0, 1.0, 5.0}`` depending on the outcome matrix above.
    """
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
        # Buyer closed a deal even though no ZOPA existed — unusual but scored
        # positively because the seller accepted below their reserve.
        return 5.0
    return 0.0


def format_reward(completion: Messages, info: dict, **kwargs) -> float:
    """Score compliance with the required action-tag format.

    Counts the fraction of buyer turns (``assistant`` messages) that contain
    a valid ``<action>...</action>`` tag.  A score of ``1.0`` means every
    turn was correctly formatted; ``0.0`` means no turn had a valid tag.

    Args:
        completion: Buyer-side chat history.
        info: Product info dict (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        A float in ``[0.0, 1.0]``.  Returns ``0.0`` if there are no buyer
        turns (e.g. the episode ended before the first step).
    """
    buyer_turns = [message for message in completion if message["role"] == "assistant"]
    if not buyer_turns:
        return 0.0
    valid = sum(
        1 for message in buyer_turns if _parse_action(message["content"])[0] != "INVALID"
    )
    return valid / len(buyer_turns)


def efficiency_bonus(completion: Messages, info: dict, **kwargs) -> float:
    """Score how quickly the buyer closed the deal.

    Rewards closing in fewer turns relative to the episode's turn budget.
    A deal closed on turn 1 of a 10-turn budget scores ``0.9``; a deal closed
    on the last turn scores ``0.0``.

    Returns ``0.0`` if no deal was reached (walking away is not rewarded for
    efficiency).

    Args:
        completion: Buyer-side chat history (unused directly).
        info: Product info dict; used as a fallback source for ``max_turns``.
        **kwargs: Must contain ``state`` (dict from ``reward_state()``).

    Returns:
        A float in ``[0.0, 1.0)``.
    """
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
    """Score the quality of the buyer's opening anchor offer.

    The ideal opening anchor is ``0.65 × buyer_true_value`` — low enough to
    leave room for concessions but not so low as to be rejected outright.
    The score decays linearly as the opening offer deviates from this ideal,
    clamped to ``[-1.0, 1.0]``.

    Returns ``0.0`` if ``buyer_true_value`` is unknown or no offers were made.

    Args:
        completion: Buyer-side chat history.
        info: Product info dict; used as a fallback source for
            ``buyer_true_value``.
        **kwargs: Must contain ``state`` (dict from ``reward_state()``).

    Returns:
        A float in ``[-1.0, 1.0]``.
    """
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
    # Ideal anchor is 65 % of the buyer's true value.
    ideal = 0.65 * buyer_true_value
    distance = abs(opening_offer - ideal) / buyer_true_value
    return float(_clamp(1.0 - 2.0 * distance, -1.0, 1.0))


def negotiation_progress_reward(
    completion: Messages, info: dict, **kwargs
) -> float:
    """Score the quality of the buyer's concession pattern across all offers.

    Good negotiation involves making small, controlled upward concessions
    (increasing offers gradually).  This function penalises two failure modes:
    - **Backtracking**: lowering an offer scores ``-1.0`` for that step.
    - **Large jumps**: a concession larger than ``1/alpha`` of ``buyer_true_value``
      scores negatively (``alpha = 4.0``, so jumps > 25 % of true value are penalised).

    The per-step scores are averaged and clamped to ``[-1.0, 1.0]``.

    Returns ``0.0`` if fewer than two offers were made (no concession pattern
    to evaluate) or if ``buyer_true_value`` is unknown.

    Args:
        completion: Buyer-side chat history.
        info: Product info dict (unused directly).
        **kwargs: Must contain ``state`` (dict from ``reward_state()``).

    Returns:
        A float in ``[-1.0, 1.0]``.
    """
    state = kwargs.get("state", {})
    buyer_true_value = state.get("buyer_true_value")
    if not buyer_true_value:
        return 0.0

    buyer_offers = _get_buyer_offers(completion)
    if len(buyer_offers) < 2:
        return 0.0

    # alpha controls how harshly large concession jumps are penalised.
    # With alpha=4.0, a jump equal to 25 % of true value scores exactly 0.0.
    alpha = 4.0
    rewards = []

    for index in range(1, len(buyer_offers)):
        prev_offer = buyer_offers[index - 1]
        curr_offer = buyer_offers[index]
        delta = curr_offer - prev_offer

        if delta < 0:
            # Buyer lowered their offer — penalise backtracking.
            rewards.append(-1.0)
            continue

        # Score the concession size: small steps score near 1.0, large steps
        # score lower (and can go negative for very large jumps).
        ratio = delta / buyer_true_value
        step_score = 1.0 - alpha * ratio
        rewards.append(step_score)

    return float(_clamp(sum(rewards) / len(rewards), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def reward_breakdown(trajectory: TrajectoryResult) -> dict[str, float]:
    """Compute all six reward component scores for a completed trajectory.

    Calls each reward function with the shared ``completion``, ``info``, and
    ``state`` arguments derived from the trajectory.

    Args:
        trajectory: The completed negotiation trajectory to evaluate.

    Returns:
        A dict mapping each component name to its raw (un-normalised) score:
        ``surplus_reward``, ``walkaway_penalty``, ``format_reward``,
        ``efficiency_bonus``, ``anchoring_reward``,
        ``negotiation_progress_reward``.
    """
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
    """Aggregate all reward components into a single score in ``[0, 1]``.

    Each component is normalised to ``[0, 1]`` using its known raw range
    before averaging:

    - ``surplus_reward``              → ``(raw + 1) / 2``
    - ``walkaway_penalty``            → ``(raw + 5) / 10``
    - ``format_reward``               → unchanged (already in [0, 1])
    - ``efficiency_bonus``            → unchanged (already in [0, 1])
    - ``anchoring_reward``            → ``(raw + 1) / 2``
    - ``negotiation_progress_reward`` → ``(raw + 1) / 2``

    The six normalised scores are then averaged with equal weight.

    Args:
        trajectory: The completed negotiation trajectory to score.

    Returns:
        A float in ``[0.0, 1.0]`` representing the overall quality of the
        buyer's negotiation behaviour.
    """
    breakdown = reward_breakdown(trajectory)
    normalized = {
        "surplus_reward": (breakdown["surplus_reward"] + 1.0) / 2.0,
        "walkaway_penalty": (breakdown["walkaway_penalty"] + 5.0) / 10.0,
        "format_reward": breakdown["format_reward"],
        "efficiency_bonus": breakdown["efficiency_bonus"],
        "anchoring_reward": (breakdown["anchoring_reward"] + 1.0) / 2.0,
        "negotiation_progress_reward": (
            breakdown["negotiation_progress_reward"] + 1.0
        ) / 2.0,
    }
    return float(sum(normalized.values()) / len(normalized))
