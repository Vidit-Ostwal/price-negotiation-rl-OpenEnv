# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Price Negotiation Environment Implementation.

This module implements the server-side ``PriceNegotiationEnvironment`` class,
which drives one full buyer-seller negotiation episode.

Episode lifecycle
-----------------
1. ``reset()`` samples a product scenario from ``dataset.json``, seeds
   separate buyer and seller chat histories with their respective system
   prompts, and returns an initial ``PriceNegotiationObservation``.
2. ``step()`` receives the buyer's natural-language response, appends it to
   both chat histories, and then either:
   - terminates immediately if the buyer used an ``ACCEPT`` or ``WALK`` tag, or
   - calls the seller LLM to generate a counter-response and checks whether
     the seller terminated the episode.
3. The episode ends when either side uses ``ACCEPT`` or ``WALK``, or when the
   caller enforces an external turn limit.

Environment variables consumed here:
    SELLER_MODEL – Model identifier for seller-side LLM generation.
                   Defaults to ``DEFAULT_OPENAI_MODEL``.
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

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

# Default model used when SELLER_MODEL is not set in the environment.
DEFAULT_OPENAI_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Model used to generate seller replies during ``step()``.
# Override via the SELLER_MODEL environment variable to swap in a different
# model without changing code (e.g. a smaller/faster model for local testing).
SELLER_MODEL = os.getenv("SELLER_MODEL", DEFAULT_OPENAI_MODEL)

# Type alias for the three supported difficulty levels.
# Used as a parameter type in ``reset()`` and ``_sample_product_info_for_difficulty()``.
Difficulty = Literal["easy", "medium", "hard"]


class PriceNegotiationEnvironment(Environment):
    """Server-side environment that runs one buyer-vs-seller negotiation episode.

    The agent (buyer) interacts with this environment through the standard
    OpenEnv ``reset`` / ``step`` / ``state`` interface.  The seller side is
    driven by an LLM via ``get_openai_response`` so that the buyer faces a
    realistic, adaptive counterpart.

    Each instance is stateful and represents a single concurrent session.
    ``SUPPORTS_CONCURRENT_SESSIONS = True`` tells the OpenEnv server factory
    to create a fresh instance per WebSocket connection rather than sharing
    one instance across all clients.

    Attributes:
        _state (PriceNegotiationState): Pydantic state object exposed to the
            client via the ``/state`` endpoint.
        _reset_count (int): Number of times ``reset()`` has been called on
            this instance.  Used to cycle deterministically through the dataset.
        _dataset (list[dict]): All negotiation scenarios loaded from
            ``dataset.json`` at construction time.
        product_info (dict | None): The scenario sampled for the current
            episode, including product metadata, valuations, and prompts.
        buyer_messages (list[dict[str, str]]): OpenAI-format chat history
            for the buyer side (system prompt + alternating assistant/user turns).
        seller_messages (list[dict[str, str]]): OpenAI-format chat history
            for the seller side (system prompt + alternating user/assistant turns).

    Example:
        >>> env = PriceNegotiationEnvironment()
        >>> obs = env.reset(difficulty="easy")
        >>> print(obs.deal_status)   # "ONGOING"
        >>> print(obs.next_turn)     # "BUYER"
        >>>
        >>> obs = env.step(PriceNegotiationAction(
        ...     buyer_response="I'd offer $500. <action>OFFER $500</action>"
        ... ))
        >>> print(obs.deal_status)   # "ONGOING" or "ACCEPTED" / "WALKED_AWAY"
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize a new price negotiation environment instance.

        Loads the full scenario dataset from disk once at construction time so
        that subsequent ``reset()`` calls are fast (no I/O).  All episode-level
        state is set to empty/default values and will be populated on the first
        ``reset()`` call.
        """
        self._state = PriceNegotiationState(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._dataset = self._load_dataset()
        self.product_info = None
        self.buyer_messages: list[dict[str, str]] = []
        self.seller_messages: list[dict[str, str]] = []

    def _refresh_state(self) -> None:
        """Sync the public ``_state`` object with the latest episode context.

        Called after every mutation of ``product_info``, ``buyer_messages``, or
        ``seller_messages`` so that the ``/state`` endpoint always returns an
        up-to-date snapshot.
        """
        self._state.product_info = self.product_info or {}
        self._state.buyer_messages = self.buyer_messages
        self._state.seller_messages = self.seller_messages

    def _load_dataset(self) -> list[dict]:
        """Load all negotiation scenarios from ``dataset.json``.

        The file is expected to live in the same directory as this module.

        Returns:
            A list of scenario dicts, each containing ``product``, ``valuations``,
            ``buyer_prompt``, ``seller_prompt``, ``metadata``, and ``difficulty``.

        Raises:
            FileNotFoundError: If ``dataset.json`` does not exist next to this file.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        dataset_path = Path(__file__).with_name("dataset.json")
        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            return json.load(dataset_file)

    def _sample_product_info(self) -> dict:
        """Pick a negotiation scenario by cycling through the full dataset.

        Uses ``_reset_count`` modulo the dataset length so that successive
        episodes rotate through all available scenarios in order.

        Returns:
            A single scenario dict from ``_dataset``.

        Raises:
            ValueError: If the dataset is empty.
        """
        if not self._dataset:
            raise ValueError("Dataset is empty")
        index = (self._reset_count - 1) % len(self._dataset)
        return self._dataset[index]

    def _sample_product_info_for_difficulty(self, difficulty: Difficulty | None) -> dict:
        """Pick a scenario matching the requested difficulty level.

        When ``difficulty`` is ``None`` the full dataset is used (no filtering).
        Otherwise only scenarios whose top-level ``difficulty`` field or
        ``valuations.difficulty`` field matches are considered, and the result
        cycles through that filtered subset.

        Args:
            difficulty: One of ``"easy"``, ``"medium"``, ``"hard"``, or ``None``
                to skip filtering.

        Returns:
            A single scenario dict whose difficulty matches the request.

        Raises:
            ValueError: If no scenarios in the dataset match the requested
                difficulty.
        """
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
        """Seed buyer and seller chat histories with their system prompts.

        Both histories start with a single ``system`` message taken from the
        sampled ``product_info``.  Subsequent turns are appended by
        ``_append_buyer_response`` and ``_append_seller_response`` during
        ``step()``.

        Must be called after ``self.product_info`` has been set.
        """
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
        """Record the buyer's turn in both conversation histories.

        The buyer's message is appended as an ``assistant`` turn in the
        buyer-side history (since the buyer LLM is the assistant there) and
        as a ``user`` turn in the seller-side history (since the seller LLM
        sees the buyer as the user).

        Args:
            buyer_response: The raw text produced by the buyer agent,
                including any action tags.
        """
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
        """Record the seller's turn in both conversation histories.

        The seller's message is appended as an ``assistant`` turn in the
        seller-side history and as a ``user`` turn in the buyer-side history
        (since the buyer LLM sees the seller as the user).

        Args:
            seller_response: The raw text produced by the seller LLM,
                including any action tags.
        """
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
        """Start a new negotiation episode.

        Samples a product scenario (optionally filtered by difficulty), seeds
        both chat histories with their system prompts, and resets all
        episode-level counters.

        Args:
            seed: Unused.  Accepted for API compatibility with the OpenEnv
                ``Environment`` interface.
            episode_id: Optional explicit episode identifier.  A random UUID
                is generated when not provided.
            difficulty: Difficulty level of the scenario to sample.  One of
                ``"easy"``, ``"medium"``, or ``"hard"``.  When ``None`` the
                environment cycles through all scenarios regardless of difficulty.
            **kwargs: Additional keyword arguments are silently ignored for
                forward compatibility.

        Returns:
            A ``PriceNegotiationObservation`` with ``deal_status="ONGOING"``,
            ``next_turn="BUYER"``, ``negotiation_round=0``, and ``done=False``,
            indicating the episode is ready for the first buyer action.
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
        """Execute one buyer turn and, if the episode continues, one seller turn.

        Processing order:
        1. Increment the step counter and append the buyer's response to both
           chat histories.
        2. Check whether the buyer used a terminal action tag:
           - ``WALK``   → episode ends with ``WALKED_AWAY``.
           - ``ACCEPT`` → episode ends with ``ACCEPTED``.
        3. If no terminal action was detected, call the seller LLM to generate
           a counter-response and append it to both histories.
        4. Check whether the seller used a terminal action tag with the same
           logic as step 2.
        5. If neither side terminated, return an ``ONGOING`` observation so the
           caller can send the next buyer turn.

        Note: Terminal-action detection uses a simple substring check (``"WALK"``
        / ``"ACCEPT"`` in the response text).  The action tags in the dataset
        prompts are ``<action>WALK</action>`` and ``<action>ACCEPT</action>``,
        so this check is intentionally broad to tolerate minor formatting
        variations from the LLM.

        Args:
            action: The buyer's next negotiation move.  ``action.buyer_response``
                must be a non-empty string and should contain exactly one action
                tag (``<action>OFFER $X</action>``, ``<action>ACCEPT</action>``,
                or ``<action>WALK</action>``).

        Returns:
            A ``PriceNegotiationObservation`` reflecting the outcome of this
            step.  ``done=True`` when the episode has ended (either side
            accepted or walked away); ``done=False`` when the negotiation is
            still ongoing.
        """
        self._state.step_count += 1
        buyer_response = action.buyer_response

        # Record the buyer's message before checking for terminal actions so
        # that the full conversation history is always consistent.
        self._append_buyer_response(buyer_response)
        self._refresh_state()

        # --- Buyer-initiated termination ---
        if "WALK" in buyer_response:
            # Buyer walked away; episode ends immediately without a seller reply.
            return PriceNegotiationObservation(
                next_turn="SELLER",
                negotiation_round=self._state.step_count,
                deal_status="WALKED_AWAY",
                done=True,
                reward=0.0,
            )

        if "ACCEPT" in buyer_response:
            # Buyer accepted the seller's last offer; episode ends immediately.
            return PriceNegotiationObservation(
                next_turn="SELLER",
                negotiation_round=self._state.step_count,
                deal_status="ACCEPTED",
                done=True,
                reward=0.0,
            )

        # --- Seller response ---
        # The buyer made an offer (or sent a non-terminal message), so ask the
        # seller LLM to generate a counter-response.
        seller_response = get_openai_response(self.seller_messages, SELLER_MODEL)
        self._append_seller_response(seller_response)
        self._refresh_state()

        # --- Seller-initiated termination ---
        if "WALK" in seller_response:
            # Seller walked away; episode ends, next_turn is BUYER for logging
            # purposes but done=True means no further steps are expected.
            return PriceNegotiationObservation(
                next_turn="BUYER",
                negotiation_round=self._state.step_count,
                deal_status="WALKED_AWAY",
                done=True,
                reward=0.0,
            )

        if "ACCEPT" in seller_response:
            # Seller accepted the buyer's offer; episode ends successfully.
            return PriceNegotiationObservation(
                next_turn="BUYER",
                negotiation_round=self._state.step_count,
                deal_status="ACCEPTED",
                done=True,
                reward=0.0,
            )

        # --- Negotiation continues ---
        # Neither side terminated; return control to the buyer for the next turn.
        return PriceNegotiationObservation(
            next_turn="BUYER",
            negotiation_round=self._state.step_count,
            deal_status="ONGOING",
            done=False,
            reward=0.0,
        )

    @property
    def state(self) -> PriceNegotiationState:
        """Return the current episode state.

        The returned object is the live ``_state`` instance, which is kept
        in sync with ``product_info``, ``buyer_messages``, and
        ``seller_messages`` via ``_refresh_state()``.  Callers that need an
        immutable snapshot should call ``state.model_copy(deep=True)``.

        Returns:
            The current ``PriceNegotiationState``, including ``episode_id``,
            ``step_count``, ``product_info``, ``buyer_messages``, and
            ``seller_messages``.
        """
        return self._state
