# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Price Negotiation Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        PriceNegotiationAction,
        PriceNegotiationObservation,
        PriceNegotiationState,
    )
    from .helper_functions import get_openai_response
except ImportError:
    from models import (
        PriceNegotiationAction,
        PriceNegotiationObservation,
        PriceNegotiationState,
    )
    from server.helper_functions import get_openai_response

import os
DEFAULT_OPENAI_MODEL = "Qwen/Qwen2.5-72B-Instruct"
SELLER_MODEL = os.getenv("SELLER_MODEL", DEFAULT_OPENAI_MODEL)
Difficulty = Literal["easy", "medium", "hard"]


class PriceNegotiationEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = PriceNegotiationEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Price Negotiation environment ready!"
        >>>
        >>> obs = env.step(PriceNegotiationAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the price_negotiation environment."""
        self._state = PriceNegotiationState(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._dataset = self._load_dataset()
        self.product_info = None
        self.buyer_messages: list[dict[str, str]] = []
        self.seller_messages: list[dict[str, str]] = []

    def _refresh_state(self) -> None:
        """Keep the internal state aligned with the latest episode context."""
        self._state.product_info = self.product_info or {}
        self._state.buyer_messages = self.buyer_messages
        self._state.seller_messages = self.seller_messages

    def _load_dataset(self) -> list[dict]:
        """Load the negotiation dataset from disk."""
        dataset_path = Path(__file__).with_name("dataset.json")
        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            return json.load(dataset_file)

    def _sample_product_info(self) -> dict:
        """Pick a negotiation scenario by cycling through the dataset."""
        if not self._dataset:
            raise ValueError("Dataset is empty")
        index = (self._reset_count - 1) % len(self._dataset)
        return self._dataset[index]

    def _sample_product_info_for_difficulty(self, difficulty: Difficulty | None) -> dict:
        """Pick a scenario matching the requested difficulty when provided."""
        if difficulty is None:
            return self._sample_product_info()

        matches = [
            item
            for item in self._dataset
            if item.get("difficulty") == difficulty
            or item.get("valuations", {}).get("difficulty") == difficulty
        ]
        if not matches:
            raise ValueError(f"Unknown or unavailable difficulty: {difficulty}")
        index = (self._reset_count - 1) % len(matches)
        return matches[index]

    def _initialize_messages(self) -> None:
        """Seed buyer and seller chat histories with system prompts."""
        self.buyer_messages = [
            {
                "role": "system",
                "content": self.product_info["buyer_prompt"],
            },
        ]
        self.seller_messages = [
            {
                "role": "system",
                "content": self.product_info["seller_prompt"],
            }
        ]

    def _append_buyer_response(self, buyer_response: str) -> None:
        """Add the buyer response to both conversation histories."""
        self.buyer_messages.append(
            {
                "role": "assistant",
                "content": buyer_response,
            }
        )
        self.seller_messages.append(
            {
                "role": "user",
                "content": buyer_response,
            }
        )

    def _append_seller_response(self, seller_response: str) -> None:
        """Add the seller response to both conversation histories."""
        self.seller_messages.append(
            {
                "role": "assistant",
                "content": seller_response,
            }
        )
        self.buyer_messages.append(
            {
                "role": "user",
                "content": seller_response,
            }
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        difficulty: Difficulty | None = None,
        **kwargs,
    ) -> PriceNegotiationObservation:
        """
        Reset the environment.

        Returns:
            PriceNegotiationObservation with a ready message
        """
        del seed, kwargs
        self._state = PriceNegotiationState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1
        self.product_info = self._sample_product_info_for_difficulty(difficulty)
        self._initialize_messages()
        self._refresh_state()

        return PriceNegotiationObservation(
            next_turn="BUYER",
            negotiation_round=0,
            deal_status="ONGOING",
            done=False,
            reward=0.0,
        )

    def step(self, action: PriceNegotiationAction) -> PriceNegotiationObservation:  # type: ignore[override]
        """
        Execute one buyer step in the negotiation.

        Args:
            action: PriceNegotiationAction for the buyer's next move

        Returns:
            PriceNegotiationObservation with the updated negotiation state
        """
        self._state.step_count += 1
        buyer_response = action.buyer_response

        self._append_buyer_response(buyer_response)
        self._refresh_state()

        if "WALK" in buyer_response:
            return PriceNegotiationObservation(
                next_turn="SELLER",
                negotiation_round=self._state.step_count,
                deal_status="WALKED_AWAY",
                done=True,
                reward=0.0,
            )

        if "ACCEPT" in buyer_response:
            return PriceNegotiationObservation(
                next_turn="SELLER",
                negotiation_round=self._state.step_count,
                deal_status="ACCEPTED",
                done=True,
                reward=0.0,
            )

        seller_response = get_openai_response(self.seller_messages, SELLER_MODEL)
        self._append_seller_response(seller_response)
        self._refresh_state()

        if "WALK" in seller_response:
            return PriceNegotiationObservation(
                next_turn="BUYER",
                negotiation_round=self._state.step_count,
                deal_status="WALKED_AWAY",
                done=True,
                reward=0.0,
            )

        if "ACCEPT" in seller_response:
            return PriceNegotiationObservation(
                next_turn="BUYER",
                negotiation_round=self._state.step_count,
                deal_status="ACCEPTED",
                done=True,
                reward=0.0,
            )

        return PriceNegotiationObservation(
            next_turn="BUYER",
            negotiation_round=self._state.step_count,
            deal_status="ONGOING",
            done=False,
            reward=0.0,
        )

    @property
    def state(self) -> PriceNegotiationState:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
