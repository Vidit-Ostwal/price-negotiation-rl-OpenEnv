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
    from .price_negotiation_environment import PriceNegotiationEnvironment
except ImportError:
    from models import PriceNegotiationAction, PriceNegotiationObservation
    from server.price_negotiation_environment import PriceNegotiationEnvironment


# Build the core OpenEnv HTTP/WS API.
app = create_fastapi_app(
    PriceNegotiationEnvironment,
    PriceNegotiationAction,
    PriceNegotiationObservation,
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Mount custom static UI
enable_web = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")
if enable_web:
    static_dir = Path(__file__).parent / "static"
    web_env = PriceNegotiationEnvironment()

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
        return serialize_observation(observation)

    @app.post("/step")
    async def web_step(payload: dict[str, Any] = Body(...)):
        """Step the persistent local web UI episode."""
        action_data = payload.get("action", payload)
        try:
            action = PriceNegotiationAction(**action_data)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors()) from e

        observation = web_env.step(action)
        return serialize_observation(observation)

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
