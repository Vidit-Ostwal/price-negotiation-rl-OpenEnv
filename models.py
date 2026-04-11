# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic data models for the Price Negotiation Environment.

This module defines the three core data-contract types shared between the
server (``PriceNegotiationEnvironment``) and the client (``PriceNegotiationEnv``):

- ``PriceNegotiationAction``      — what the buyer sends each turn
- ``PriceNegotiationObservation`` — what the buyer receives after each turn
- ``PriceNegotiationState``       — the full internal episode state (exposed
                                    via ``GET /state``)

All three extend the corresponding OpenEnv base types (``Action``,
``Observation``, ``State``) so they are automatically compatible with the
OpenEnv HTTP/WebSocket server and client infrastructure.

The agent always plays the **buyer** role.  Each action is the buyer's next
natural-language negotiation move (which must include one action tag), and
each observation is the seller's response together with the updated
negotiation metadata visible to the buyer.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class PriceNegotiationAction(Action):
    """A single buyer turn in the negotiation.

    The buyer's response must be natural language that ends with exactly one
    of the following action tags so the environment can parse the intent:

    - ``<action>OFFER $X</action>``  — make or counter with a specific price
    - ``<action>ACCEPT</action>``    — accept the seller's current offer
    - ``<action>WALK</action>``      — walk away from the negotiation

    The text before the tag may include reasoning, counter-arguments, or
    market references.  The environment uses a substring check for the tag,
    so the tag must appear verbatim (case-insensitive for OFFER/ACCEPT/WALK).

    Example::

        PriceNegotiationAction(
            buyer_response=(
                "That price is too high given current market comps. "
                "I can do $450. <action>OFFER $450</action>"
            )
        )
    """

    buyer_response: str = Field(
        ...,
        description=(
            "The buyer's natural-language response to the seller, including "
            "exactly one action tag: <action>OFFER $X</action>, "
            "<action>ACCEPT</action>, or <action>WALK</action>."
        ),
    )


class PriceNegotiationObservation(Observation):
    """What the buyer agent observes after each step.

    Returned by ``env.step()`` and embedded in ``StepResult``.  Inherits
    ``done`` and ``reward`` from the OpenEnv ``Observation`` base class.

    The three fields below capture the negotiation-specific metadata that
    the buyer needs to decide its next move.  The seller's actual reply text
    is not included here — it is appended to ``buyer_messages`` in the
    environment state and can be retrieved via ``env.state()``.
    """

    next_turn: Literal["BUYER", "SELLER"] = Field(
        default="BUYER",
        description=(
            "Whose turn it is next.  Always ``'BUYER'`` when ``done=False`` "
            "(the agent must act again).  Set to ``'SELLER'`` on terminal "
            "steps for logging purposes."
        ),
    )
    negotiation_round: int = Field(
        default=0,
        description=(
            "The step number at which this observation was produced, "
            "starting from 1 after the first ``step()`` call.  Useful for "
            "enforcing turn budgets on the client side."
        ),
    )
    deal_status: Literal["ONGOING", "ACCEPTED", "WALKED_AWAY"] = Field(
        default="ONGOING",
        description=(
            "Current outcome of the negotiation.  ``'ONGOING'`` while the "
            "episode is still active; ``'ACCEPTED'`` when either side "
            "accepted an offer; ``'WALKED_AWAY'`` when either side walked."
        ),
    )


class PriceNegotiationState(State):
    """Full internal state of a negotiation episode.

    Returned by ``env.state()`` (``GET /state``).  Inherits ``episode_id``
    and ``step_count`` from the OpenEnv ``State`` base class.

    This object gives the buyer agent (and reward functions) access to the
    complete conversation histories and the sampled product scenario,
    including private valuation data that is not visible in the observation.

    Note:
        ``buyer_messages`` and ``seller_messages`` are in OpenAI chat format
        (list of ``{"role": ..., "content": ...}`` dicts).  The buyer history
        uses ``"assistant"`` for buyer turns and ``"user"`` for seller turns;
        the seller history uses the opposite convention.
    """

    product_info: dict = Field(
        default_factory=dict,
        description=(
            "The full scenario dict sampled from ``dataset.json`` for this "
            "episode.  Contains ``product``, ``valuations`` (including "
            "``buyer_true_value``, ``seller_reserve_price``, ``zopa_width``, "
            "``deal_possible``), ``metadata`` (``max_turns``, currency, "
            "behavioral descriptors), and the raw ``buyer_prompt`` / "
            "``seller_prompt`` strings."
        ),
    )
    buyer_messages: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "OpenAI-format chat history from the buyer's perspective.  "
            "Starts with a ``system`` message (the buyer prompt) and grows "
            "by two messages per round: an ``assistant`` turn (buyer's "
            "response) followed by a ``user`` turn (seller's reply).  "
            "Pass this list directly to an OpenAI-compatible API to generate "
            "the next buyer turn."
        ),
    )
    seller_messages: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "OpenAI-format chat history from the seller's perspective.  "
            "Starts with a ``system`` message (the seller prompt) and grows "
            "by two messages per round: a ``user`` turn (buyer's message) "
            "followed by an ``assistant`` turn (seller's reply).  "
            "Used internally by the environment to generate seller responses."
        ),
    )
