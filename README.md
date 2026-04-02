---
title: Price Negotiation Environment Server
emoji: 🎭
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Price Negotiation Environment

An OpenEnv negotiation environment where the agent plays the buyer and negotiates against a seller over a sampled marketplace listing. The server exposes standard OpenEnv HTTP and WebSocket endpoints and is configured to deploy as a Hugging Face Space.

The negotiation scenarios live in `server/dataset.json`. On each reset, the environment samples one scenario, seeds private buyer and seller prompts, and uses an OpenAI model to generate seller replies.

## What This Repo Contains

- A FastAPI/OpenEnv server in `server/app.py`
- The environment implementation in `server/price_negotiation_environment.py`
- Client and schema models in `client.py` and `models.py`
- Rollout and reward utilities in `rollout.py`, `reward.py`, and `inference.py`

## Environment Model

Each episode samples a product listing with:

- Product metadata and market price
- Private buyer and seller valuations
- Negotiation-style instructions for both sides
- A maximum turn budget in episode metadata

The buyer agent must respond with natural language that includes one of these action tags:

```text
<action>OFFER $X</action>
<action>ACCEPT</action>
<action>WALK</action>
```

The environment ends when either side accepts or walks away.

## Requirements

- Python 3.10+
- `uv` for local dependency management
- `OPENAI_API_KEY` set in the environment

This repo depends on `openenv-core` and `openai`, as defined in `pyproject.toml`.

## Local Setup

Install dependencies:

```bash
uv sync
```

Set your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

Run the server locally:

```bash
uv run --project . server
```

Or run Uvicorn directly:

```bash
uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

The default local server URL is `http://localhost:8000`.

## Using the Client

```python
from price_negotiation import PriceNegotiationAction, PriceNegotiationEnv

with PriceNegotiationEnv(base_url="http://localhost:8000").sync() as env:
    reset_result = env.reset()
    print(reset_result.observation)

    step_result = env.step(
        PriceNegotiationAction(
            buyer_response="I can do $450. <action>OFFER $450</action>"
        )
    )
    print(step_result.observation)

    state = env.state()
    print(state.product_info["product"]["name"])
```

The client sends `buyer_response` as the action payload and returns observations with:

- `next_turn`
- `negotiation_round`
- `deal_status`
- `done`
- `reward`

The environment state also exposes:

- `product_info`
- `buyer_messages`
- `seller_messages`
- `episode_id`
- `step_count`

## Running a Full Rollout

`inference.py` runs a complete buyer-side rollout against a live server and then scores the resulting trajectory.

Example:

```bash
uv run python inference.py --base-url http://localhost:8000 --buyer-model gpt-4.1-mini --verbose
```

This script:

- Connects to the running negotiation server
- Generates buyer turns with an OpenAI model
- Captures the final negotiation trajectory
- Prints a reward breakdown and final trajectory score

Reward helpers in `reward.py` include components such as deal outcome, surplus, formatting compliance, efficiency, anchoring, and concession quality.

## API Surface

The OpenEnv server provides the usual endpoints, including:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /health`
- `WS /ws`

When deployed in Hugging Face Spaces, the web UI is served at `/web` and the app runs on port `8000`.

## Docker

The repo includes `server/Dockerfile` for local builds and Hugging Face Spaces deployment.

Build locally with:

```bash
docker build -t price_negotiation-env:latest -f server/Dockerfile .
```

Run locally with:

```bash
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY price_negotiation-env:latest
```

## Hugging Face Spaces

This repository is already configured for Hugging Face Spaces through the YAML frontmatter above and `openenv.yaml`. Keep those deployment settings intact.

If you are deploying through OpenEnv tooling, use:

```bash
openenv push
```

That uses the environment definition in `openenv.yaml` and the Docker-based Space configuration already present in this repo.

## Project Structure

```text
.
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── reward.py
├── rollout.py
├── trajectory_types.py
└── server
    ├── app.py
    ├── dataset.json
    ├── Dockerfile
    ├── helper_functions.py
    ├── price_negotiation_environment.py
    └── requirements.txt
```
