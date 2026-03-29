# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Price Negotiation Environment.

The agent acts as the buyer. Each action is the buyer's next negotiation move,
and each observation is the seller's response together with the updated
negotiation state visible to the buyer.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class PriceNegotiationAction(Action):
    """Action taken by the buyer agent in the negotiation."""

    buyer_response: str = Field(
        ...,
        description="The buyer's natural-language response to the seller",
    )


class PriceNegotiationObservation(Observation):
    """Observation returned by the environment after the buyer acts."""

    next_turn: Literal["BUYER", "SELLER"] = Field(
        default="BUYER",
        description="Whose turn it is next, such as BUYER or SELLER",
    )
    negotiation_round: int = Field(
        default=0,
        description="The current turn number in the negotiation",
    )
    deal_status: Literal["ONGOING", "ACCEPTED", "WALKED_AWAY"] = Field(
        default="ONGOING",
        description="Negotiation status such as ONGOING, ACCEPTED, or WALKED_AWAY",
    )


class PriceNegotiationState(State):
    """Internal environment state for a negotiation episode."""

    product_info: dict = Field(
        default_factory=dict,
        description="The sampled product and negotiation scenario for the episode",
    )
    buyer_messages: list[dict[str, str]] = Field(
        default_factory=list,
        description="Buyer-side chat history",
    )
    seller_messages: list[dict[str, str]] = Field(
        default_factory=list,
        description="Seller-side chat history",
    )
