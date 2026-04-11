# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Price Negotiation Environment Client.

This module exposes ``PriceNegotiationEnv``, the typed Python client for the
price negotiation server.  It extends the generic OpenEnv ``EnvClient`` with
three concrete methods that handle serialisation and deserialisation of the
domain-specific action, observation, and state types.

Typical usage
-------------
Synchronous (blocking) style::

    from price_negotiation import PriceNegotiationAction, PriceNegotiationEnv

    with PriceNegotiationEnv(base_url="http://localhost:8000").sync() as env:
        reset_result = env.reset(difficulty="easy")
        step_result  = env.step(
            PriceNegotiationAction(
                buyer_response="I can do $450. <action>OFFER $450</action>"
            )
        )
        state = env.state()

Asynchronous style::

    async with PriceNegotiationEnv(base_url="http://localhost:8000") as env:
        reset_result = await env.reset()
        step_result  = await env.step(PriceNegotiationAction(buyer_response="..."))
        state        = await env.state()
"""

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
    """Typed client for the Price Negotiation Environment server.

    Wraps the generic ``EnvClient`` WebSocket/HTTP transport with
    domain-specific serialisation logic for the three negotiation types:
    ``PriceNegotiationAction``, ``PriceNegotiationObservation``, and
    ``PriceNegotiationState``.

    Each client instance maintains its own persistent WebSocket connection to
    the server, giving it a dedicated environment session with isolated state.
    This means multiple clients can run concurrent episodes against the same
    server without interfering with each other (provided the server was started
    with ``max_concurrent_envs > 1``).

    The client can be used as an async context manager (default) or wrapped
    with ``.sync()`` for blocking usage in non-async code.

    Example — synchronous:
        >>> with PriceNegotiationEnv(base_url="http://localhost:8000").sync() as env:
        ...     reset_result = env.reset(difficulty="easy")
        ...     print(reset_result.observation.deal_status)   # "ONGOING"
        ...
        ...     step_result = env.step(
        ...         PriceNegotiationAction(
        ...             buyer_response="I'd offer $450. <action>OFFER $450</action>"
        ...         )
        ...     )
        ...     print(step_result.observation.deal_status)    # "ONGOING" or terminal
        ...     state = env.state()
        ...     print(state.product_info["product"]["name"])

    Example — Docker-backed (starts container automatically):
        >>> client = PriceNegotiationEnv.from_docker_image("price_negotiation-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(
        ...         PriceNegotiationAction(buyer_response="<action>WALK</action>")
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: PriceNegotiationAction) -> Dict:
        """Serialise a ``PriceNegotiationAction`` into a JSON-ready dict.

        Called by the base ``EnvClient`` before sending a ``step`` message over
        the WebSocket.  Only the fields that the server's ``/step`` endpoint
        expects are included.

        Args:
            action: The buyer's next negotiation move.

        Returns:
            A dict with a single key ``"buyer_response"`` containing the
            buyer's natural-language text (including any action tag).
        """
        return {
            "buyer_response": action.buyer_response,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PriceNegotiationObservation]:
        """Deserialise a server step-response into a typed ``StepResult``.

        Called by the base ``EnvClient`` after receiving the server's JSON
        response to a ``step`` message.  The server wraps the observation
        fields inside an ``"observation"`` sub-dict, while ``"done"`` and
        ``"reward"`` live at the top level.

        Args:
            payload: Raw JSON dict returned by the server for a step request.
                Expected keys: ``"observation"`` (dict), ``"done"`` (bool),
                ``"reward"`` (float | None).

        Returns:
            A ``StepResult[PriceNegotiationObservation]`` with the parsed
            observation, reward, and done flag.
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
        """Deserialise a server state-response into a typed ``PriceNegotiationState``.

        Called by the base ``EnvClient`` after receiving the server's JSON
        response to a ``GET /state`` request.  All fields default to safe
        empty values so that callers can safely access them even if the server
        omits optional keys.

        Args:
            payload: Raw JSON dict returned by the server for a state request.
                Expected keys: ``"episode_id"`` (str | None),
                ``"step_count"`` (int), ``"product_info"`` (dict),
                ``"buyer_messages"`` (list), ``"seller_messages"`` (list).

        Returns:
            A fully populated ``PriceNegotiationState`` reflecting the current
            server-side episode context.
        """
        return PriceNegotiationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            product_info=payload.get("product_info", {}),
            buyer_messages=payload.get("buyer_messages", []),
            seller_messages=payload.get("seller_messages", []),
        )
