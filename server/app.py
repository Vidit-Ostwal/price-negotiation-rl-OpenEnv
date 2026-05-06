# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Price Negotiation Environment.

This module creates an HTTP server that exposes the PriceNegotiationEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
from pathlib import Path
from typing import Any

from fastapi import Body, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

try:
    from openenv.core.env_server.serialization import serialize_observation
    from openenv.core.env_server.http_server import create_fastapi_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import PriceNegotiationAction, PriceNegotiationObservation
    from ..reward import reward_breakdown, score_trajectory
    from ..trajectory_types import TrajectoryResult, TrajectoryStep
    from .price_negotiation_environment import PriceNegotiationEnvironment
except ImportError:
    from models import PriceNegotiationAction, PriceNegotiationObservation
    from reward import reward_breakdown, score_trajectory
    from trajectory_types import TrajectoryResult, TrajectoryStep
    from server.price_negotiation_environment import PriceNegotiationEnvironment


# Build the core OpenEnv HTTP/WS API.
app = create_fastapi_app(
    PriceNegotiationEnvironment,
    PriceNegotiationAction,
    PriceNegotiationObservation,
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

REWARD_COMPONENT_WEIGHTS = {
    "surplus_reward": 1.0 / 6.0,
    "walkaway_penalty": 1.0 / 6.0,
    "format_reward": 1.0 / 6.0,
    "efficiency_bonus": 1.0 / 6.0,
    "anchoring_reward": 1.0 / 6.0,
    "negotiation_progress_reward": 1.0 / 6.0,
}

# Mount custom static UI
enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
if enable_web:
    static_dir = Path(__file__).parent / "static"
    web_env = PriceNegotiationEnvironment()
    web_trajectory: dict[str, Any] = {
        "initial_observation": None,
        "steps": [],
    }

    def remove_route(path: str, method: str) -> None:
        """Replace OpenEnv's stateless HTTP route with the web UI's session route."""
        app.router.routes = [
            route
            for route in app.router.routes
            if not (
                getattr(route, "path", None) == path
                and method in (getattr(route, "methods", None) or set())
            )
        ]

    def latest_seller_reply() -> str | None:
        """Return the latest seller assistant reply for trajectory scoring."""
        for message in reversed(web_env.state.seller_messages):
            if message.get("role") == "assistant":
                return message.get("content", "")
        return None

    def build_web_trajectory() -> TrajectoryResult:
        """Build the current web UI trajectory from accumulated step snapshots."""
        initial_observation = web_trajectory["initial_observation"]
        if initial_observation is None:
            initial_observation = PriceNegotiationObservation(
                next_turn="BUYER",
                negotiation_round=0,
                deal_status="ONGOING",
                done=False,
                reward=0.0,
            )

        return TrajectoryResult(
            episode_id=web_env.state.episode_id,
            initial_observation=initial_observation,
            final_state=web_env.state.model_copy(deep=True),
            steps=web_trajectory["steps"],
        )

    def aggregate_reward_component(name: str, raw_score: float) -> float:
        """Convert a raw component to the [-1, 1] aggregate scale."""
        if name == "walkaway_penalty":
            return raw_score / 5.0
        return raw_score

    def build_reward_components(
        breakdown: dict[str, float] | None,
    ) -> dict[str, dict[str, float]] | None:
        """Attach aggregate-scale scores, weights, and weighted contribution."""
        if breakdown is None:
            return None

        return {
            name: {
                "raw": raw_score,
                "score": aggregate_reward_component(name, raw_score),
                "weight": REWARD_COMPONENT_WEIGHTS[name],
                "weighted_score": (
                    aggregate_reward_component(name, raw_score)
                    * REWARD_COMPONENT_WEIGHTS[name]
                ),
            }
            for name, raw_score in breakdown.items()
        }

    def serialize_with_reward(
        observation: PriceNegotiationObservation,
        reward: float,
        breakdown: dict[str, float] | None,
    ) -> dict[str, Any]:
        """Serialize an observation and attach web UI reward details."""
        observation.reward = reward
        observation.reward_breakdown = breakdown
        observation.reward_weights = REWARD_COMPONENT_WEIGHTS if breakdown else None
        observation.reward_components = build_reward_components(breakdown)
        response = serialize_observation(observation)
        reward_components = observation.reward_components
        response["observation"]["reward"] = reward
        response["observation"]["reward_breakdown"] = breakdown
        response["observation"]["reward_weights"] = observation.reward_weights
        response["observation"]["reward_components"] = reward_components
        response["reward"] = reward
        response["reward_breakdown"] = breakdown
        response["reward_weights"] = observation.reward_weights
        response["reward_components"] = reward_components
        return response

    remove_route("/reset", "POST")
    remove_route("/step", "POST")
    remove_route("/state", "GET")

    @app.post("/reset")
    async def web_reset(
        payload: dict[str, Any] | None = Body(default=None),
        difficulty: str | None = None,
    ):
        """Reset the persistent local web UI episode."""
        reset_kwargs = payload or {}
        if difficulty and difficulty != "any":
            reset_kwargs["difficulty"] = difficulty

        observation = web_env.reset(**reset_kwargs)
        web_trajectory["initial_observation"] = observation
        web_trajectory["steps"] = []
        return serialize_with_reward(observation, 0.0, None)

    @app.post("/step")
    async def web_step(payload: dict[str, Any] = Body(...)):
        """Step the persistent local web UI episode."""
        action_data = payload.get("action", payload)
        try:
            action = PriceNegotiationAction(**action_data)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors()) from e

        observation = web_env.step(action)
        web_trajectory["steps"].append(
            TrajectoryStep(
                buyer_response=action.buyer_response,
                observation=observation,
                state=web_env.state.model_copy(deep=True),
                seller_reply=latest_seller_reply(),
            )
        )
        trajectory = build_web_trajectory()
        score = score_trajectory(trajectory)
        breakdown = reward_breakdown(trajectory)
        return serialize_with_reward(observation, score, breakdown)

    @app.get("/state")
    async def web_state():
        """Return the persistent local web UI state, including product_info."""
        return web_env.state.model_dump()

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Serve custom HTML UI at root."""
        return FileResponse(static_dir / "index.html")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m price_negotiation.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn price_negotiation.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", type=int, default=8000)
    # args = parser.parse_args()
    # main(port=args.port)
