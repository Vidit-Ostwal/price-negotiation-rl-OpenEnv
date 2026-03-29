# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Price Negotiation Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    PriceNegotiationAction,
    PriceNegotiationObservation,
    PriceNegotiationState,
)


class PriceNegotiationEnv(
    EnvClient[
        PriceNegotiationAction,
        PriceNegotiationObservation,
        PriceNegotiationState,
    ]
):
    """
    Client for the Price Negotiation Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with PriceNegotiationEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(PriceNegotiationAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = PriceNegotiationEnv.from_docker_image("price_negotiation-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(PriceNegotiationAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: PriceNegotiationAction) -> Dict:
        """
        Convert PriceNegotiationAction to JSON payload for step message.

        Args:
            action: PriceNegotiationAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "buyer_response": action.buyer_response,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PriceNegotiationObservation]:
        """
        Parse server response into StepResult[PriceNegotiationObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with PriceNegotiationObservation
        """
        obs_data = payload.get("observation", {})
        observation = PriceNegotiationObservation(
            next_turn=obs_data.get("next_turn", "BUYER"),
            negotiation_round=obs_data.get("negotiation_round", 0),
            deal_status=obs_data.get("deal_status", "ONGOING"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult[PriceNegotiationObservation](
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> PriceNegotiationState:
        """
        Parse server response into PriceNegotiationState object.

        Args:
            payload: JSON response from state request

        Returns:
            PriceNegotiationState object with negotiation context
        """
        return PriceNegotiationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            product_info=payload.get("product_info", {}),
            buyer_messages=payload.get("buyer_messages", []),
            seller_messages=payload.get("seller_messages", []),
        )
