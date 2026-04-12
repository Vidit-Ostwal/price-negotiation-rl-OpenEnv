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

# 🎭 Price Negotiation Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) RL environment where an LLM agent plays the **buyer** and negotiates against an LLM-powered **seller** over real marketplace listings. The server exposes standard OpenEnv HTTP and WebSocket endpoints and is pre-configured for Hugging Face Spaces deployment.

---

## 🧠 Real-World Utility

Price negotiation is one of the most economically significant and cognitively demanding tasks humans perform. It is also one of the hardest for LLMs — requiring **multi-turn strategic reasoning**, **information asymmetry management**, and **adaptive decision-making** under uncertainty.

### What makes this environment realistic

Every episode is grounded in real negotiation economics:

- **Zone of Possible Agreement (ZOPA)** — each scenario has a `buyer_true_value` and a `seller_reserve_price`. A deal is only possible when these overlap. The agent must discover this boundary through negotiation, not by reading it.
- **Private information** — the buyer knows its own maximum willingness to pay but not the seller's floor, and vice versa. This mirrors real-world information asymmetry.
- **Adaptive adversary** — the seller is an LLM with its own system prompt, negotiation style, and private reserve price. It responds dynamically to the buyer's strategy, creating a non-stationary opponent.
- **Realistic negotiation dynamics** — scenarios encode `haggle_norm`, `typical_discount_pct`, and per-side behavioral instructions (tone, urgency, concession strategy, information strategy) drawn from real marketplace negotiation research.
- **Walk-away discipline** — the agent must learn when *not* to deal. If no ZOPA exists, the correct action is to walk away. Accepting a bad deal is penalised more than walking.

### Why training on this transfers to the real world

| Skill trained | Real-world application |
|---|---|
| Anchoring low and making controlled concessions | Salary negotiation, procurement, real estate |
| Recognising when a deal is impossible | Contract negotiation, vendor selection |
| Structured output under pressure | Any tool-use or function-calling scenario |
| Adaptive strategy against an unknown opponent | Sales, diplomacy, competitive bidding |
| Balancing speed vs. price (efficiency vs. surplus) | Time-sensitive deals, auction dynamics |

---

## 📊 Task Progression: Easy → Medium → Hard

The three difficulty levels are designed so that **easy is solvable by a capable model**, **medium requires genuine strategy**, and **hard challenges frontier models**.

| | **Easy** | **Medium** | **Hard** |
|---|---|---|---|
| **Product** | Apple MacBook Pro 16-inch 2019 | 2018 MacBook Pro 13-inch | Sony WH-1000XM4 Headphones |
| **Category** | Electronics | Electronics | Electronics |
| **Market price** | $1,820 | $1,020 | $230 |
| **Buyer true value** | $1,350 | $720 | $150 |
| **Seller reserve** | $870 | $620 | $160 |
| **ZOPA** | $870 – $1,350 | $620 – $720 | ❌ None (`deal_possible = false`) |
| **ZOPA width** | $480 (wide) | $100 (narrow) | $0 |
| **Suggested anchor** | $1,010 | $480 | $90 |
| **Buyer tone** | Polite but skeptical | Friendly and conversational | Friendly and conversational |
| **Seller tone** | Concise and professional | Concise and professional | Confident and premium-positioned |
| **Seller urgency** | High — wants to close fast | Medium — won't sell cheap | Medium — holds firm early |
| **Seller concession style** | One pragmatic closing number | Small reciprocal concessions only | Holds firm early, concedes only near closing |
| **Deal possible?** | ✅ Yes | ✅ Yes | ❌ No |

### What makes each level genuinely harder

**Easy** — Wide ZOPA ($480) gives the buyer plenty of room to manoeuvre. The seller is motivated to close quickly and will make one pragmatic offer. A capable model that anchors near $1,010 and makes controlled concessions should close a deal comfortably.

**Medium** — The ZOPA collapses to just $100 ($620–$720). The buyer must anchor precisely near $480, make only small step-by-step increases, and read the seller's signals carefully. One large concession jump can overshoot the ZOPA and leave surplus on the table. The seller makes only small reciprocal concessions, so patience and discipline are required.

**Hard** — There is **no ZOPA**. The buyer's maximum ($150) is below the seller's reserve ($160). The correct action is to **walk away** — but the seller is confident, premium-positioned, and will test the buyer's seriousness before moving. A model that doesn't recognise the impossible deal and accepts anyway is penalised heavily by `walkaway_penalty`. This tests whether the agent has learned walk-away discipline, not just deal-closing.

The grader uses an LLM to score each trajectory 0.0–1.0, making it robust to paraphrasing while still being deterministic given the same trajectory.

---

## 🚀 Deployment

### Option 1 — Hugging Face Space (recommended)

This repo is already wired for HF Spaces via the YAML frontmatter above and `openenv.yaml`. To push:

```bash
openenv push
```

The Space runs on port `8000`. The web UI is served at `/web` and the API is available at the root.

> **Keep the YAML frontmatter and `openenv.yaml` intact** — they control the Space configuration.

---

### Option 2 — Docker (local)

Build the image:

```bash
docker build -t price_negotiation-env:latest -f Dockerfile .
```

Run it (HF token only — no OpenAI key needed):

```bash
docker run --rm -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e SELLER_MODEL="Qwen/Qwen2.5-72B-Instruct" \
  price_negotiation-env:latest
```

Server is now available at `http://localhost:8000`.

---

### Option 3 — Local (uv)

```bash
# 1. Install dependencies
uv sync

# 2. Set credentials
export HF_TOKEN=your_hf_token_here
export API_BASE_URL="https://router.huggingface.co/v1"   # or your own endpoint
export SELLER_MODEL="Qwen/Qwen2.5-72B-Instruct"          # optional, this is the default

# 3. Start the server
uv run --project . server
# or
uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

---

## 🖥️ Environment Design

### How the server works

The server is a **FastAPI** app created by OpenEnv's `create_app()` factory (`server/app.py`). It wraps `PriceNegotiationEnvironment` and exposes it over HTTP and WebSocket. Each WebSocket client gets its own isolated environment instance (`SUPPORTS_CONCURRENT_SESSIONS = True`).

### Episode lifecycle

`reset()` produces a **fully clean state** — new episode UUID, zeroed step count, freshly sampled scenario, and newly seeded chat histories. No state leaks between episodes.

```
env.reset(difficulty?)
    └─ sample scenario from dataset.json (cycles deterministically by reset count)
    └─ seed buyer system prompt  →  buyer_messages = [{"role": "system", ...}]
    └─ seed seller system prompt →  seller_messages = [{"role": "system", ...}]
    └─ return PriceNegotiationObservation(deal_status=ONGOING, round=0, done=False)

env.step(buyer_response)
    ├─ append buyer turn to both histories
    ├─ buyer says WALK   → WALKED_AWAY, done=True  (no seller call)
    ├─ buyer says ACCEPT → ACCEPTED,    done=True  (no seller call)
    └─ call seller LLM with seller_messages
        ├─ seller says WALK   → WALKED_AWAY, done=True
        ├─ seller says ACCEPT → ACCEPTED,    done=True
        └─ append seller reply to both histories
           return ONGOING, done=False  →  buyer acts again
```

### Action space

Every buyer response **must** contain exactly one structured action tag:

```
<action>OFFER $X</action>   — make or counter with a specific dollar price
<action>ACCEPT</action>     — accept the seller's current offer on the table
<action>WALK</action>       — walk away from the negotiation entirely
```

Natural language reasoning may appear before the tag. The environment uses a regex parser (`ACTION_RE`) to extract the action — tolerant of minor formatting variations.

### Observation space

| Field | Type | Description |
|-------|------|-------------|
| `deal_status` | `ONGOING` \| `ACCEPTED` \| `WALKED_AWAY` | Current negotiation outcome |
| `negotiation_round` | `int` | Step number (starts at 1) |
| `next_turn` | `BUYER` \| `SELLER` | Who acts next |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Always `0.0` during episode; use `score_trajectory()` for final score |

### State (via `GET /state`)

| Field | Description |
|-------|-------------|
| `product_info` | Full scenario: product metadata, private valuations (`buyer_true_value`, `seller_reserve_price`, `zopa_width`, `deal_possible`), metadata (`max_turns`), and both system prompts |
| `buyer_messages` | OpenAI-format chat history from the buyer's perspective (pass directly to LLM API) |
| `seller_messages` | OpenAI-format chat history from the seller's perspective |
| `episode_id` | UUID for this episode |
| `step_count` | Number of steps taken |

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode (accepts `difficulty` in body) |
| `POST` | `/step` | Send buyer action, get observation |
| `GET` | `/state` | Full episode state |
| `GET` | `/schema` | Action / observation JSON schemas |
| `GET` | `/health` | Health check |
| `WS` | `/ws` | Persistent WebSocket session (used by the Python client) |

---

## 🤝 Client

`PriceNegotiationEnv` is a typed Python client that wraps the WebSocket transport. Each instance gets its own isolated server session.

### Synchronous usage

```python
from price_negotiation import PriceNegotiationAction, PriceNegotiationEnv

with PriceNegotiationEnv(base_url="http://localhost:8000").sync() as env:
    # Start a new episode
    reset_result = env.reset(difficulty="easy")
    print(reset_result.observation.deal_status)   # "ONGOING"

    # Take a buyer turn
    step_result = env.step(
        PriceNegotiationAction(
            buyer_response="That seems high given market comps. I'd offer $450. <action>OFFER $450</action>"
        )
    )
    print(step_result.observation.deal_status)    # "ONGOING" / "ACCEPTED" / "WALKED_AWAY"
    print(step_result.done)                       # False / True

    # Inspect full state — seller's reply is in buyer_messages[-1]
    state = env.state()
    print(state.product_info["product"]["name"])
    print(state.buyer_messages[-1]["content"])    # seller's last reply
```

### Async usage

```python
async with PriceNegotiationEnv(base_url="http://localhost:8000") as env:
    reset_result = await env.reset(difficulty="medium")
    step_result  = await env.step(PriceNegotiationAction(buyer_response="..."))
    state        = await env.state()
```

---

## 🏆 Reward Functions

All reward logic lives in `reward.py`. Scores are computed **offline** from a completed `TrajectoryResult` — the environment returns `reward=0.0` during the episode so the agent cannot exploit step-level signals.

Each reward function receives the **full buyer-side chat history** (the entire negotiation trace) and the **product info dict** (including private valuations) so it evaluates the complete episode, not just the final step.

---

### 1. `surplus_reward` — Economic outcome · `[-1, 1]`

Did the buyer get a good price? Measures how much of the ZOPA the buyer captured.

```
surplus = (buyer_true_value − final_price) / zopa_width

 1.0  → buyer paid the seller's reserve price (maximum possible surplus)
 0.0  → buyer paid exactly their own true value (zero surplus, break-even)
-1.0  → buyer paid above their true value (overpaid)
```

Returns `0.0` if no deal was reached or the final price cannot be inferred from the trace.

---

### 2. `walkaway_penalty` — Decision correctness · `{-5, 1, 5}`

Was the deal/walk decision economically rational? This is the **highest-weight component** — a missed deal when a ZOPA existed is penalised heavily.

| ZOPA exists? | Deal reached? | Score | Reason |
|---|---|---|---|
| ✅ Yes | ✅ Yes | `+1.0` | Correct — closed a profitable deal |
| ✅ Yes | ❌ No | `−5.0` | Wrong — walked away from value on the table |
| ❌ No | ❌ No | `+1.0` | Correct — recognised an impossible deal and walked |
| ❌ No | ✅ Yes | `+5.0` | Rare bonus — seller accepted below their reserve |

The `−5.0` penalty dominates the final score, making deal completion the primary training signal when a ZOPA exists.

---

### 3. `format_reward` — Structured output compliance · `[0, 1]`

Scans every buyer turn in the full trace and counts how many contain a valid `<action>` tag.

```
score = valid_turns / total_buyer_turns
```

`1.0` means every turn was correctly formatted. Directly trains the model to follow output constraints consistently across the entire episode.

---

### 4. `efficiency_bonus` — Speed of closing · `[0, 1]`

Rewards closing the deal in fewer turns relative to the episode's turn budget (`max_turns = 10`).

```
score = (max_turns − turns_used) / max_turns

Example (max_turns = 10):
  closed on turn 2  →  (10 − 2) / 10  =  0.80
  closed on turn 8  →  (10 − 8) / 10  =  0.20
  closed on turn 10 →  (10 − 10) / 10 =  0.00
```

Returns `0.0` if no deal was reached. Encourages the agent to be decisive rather than stalling.

---

### 5. `anchoring_reward` — Opening offer quality · `[-1, 1]`

Evaluates the buyer's **first offer** against the ideal anchor. Research shows that opening low (but not insultingly low) anchors the negotiation in the buyer's favour. The ideal anchor is **65% of `buyer_true_value`**.

```
ideal    = 0.65 × buyer_true_value
distance = |opening_offer − ideal| / buyer_true_value
score    = clamp(1.0 − 2.0 × distance, −1.0, 1.0)
```

A perfect anchor scores `1.0`. Opening at `buyer_true_value` (no room for concessions) scores around `−0.7`. This teaches the model to anchor strategically from the very first turn.

---

### 6. `negotiation_progress_reward` — Concession quality · `[-1, 1]`

Evaluates the **pattern of all offers** across the full trace. Good negotiation means making small, controlled upward concessions — not backtracking or making large jumps that signal desperation.

For each consecutive pair of offers `(prev, curr)` in the trace:

```
delta = curr − prev

if delta < 0:
    step_score = −1.0              # backtracking — penalised hard

else:
    ratio      = delta / buyer_true_value
    step_score = 1.0 − 4.0 × ratio
    # small step (ratio → 0)       →  score near +1.0
    # jump > 25% of true value     →  score goes negative
```

Per-step scores are averaged and clamped to `[-1, 1]`. Returns `0.0` if fewer than two offers were made.

---

### Aggregation — `score_trajectory()`

Each component is normalised to `[0, 1]` and averaged with equal weight:

```
surplus_reward              → (raw + 1) / 2
walkaway_penalty            → (raw + 5) / 10
format_reward               → unchanged  (already [0, 1])
efficiency_bonus            → unchanged  (already [0, 1])
anchoring_reward            → (raw + 1) / 2
negotiation_progress_reward → (raw + 1) / 2

final_score = mean of all six  ∈ [0, 1]
```

```python
from price_negotiation.reward import reward_breakdown, score_trajectory

breakdown = reward_breakdown(trajectory)
# {
#   'surplus_reward': 0.83,        # captured 83% of ZOPA
#   'walkaway_penalty': 1.0,       # correct deal decision
#   'format_reward': 1.0,          # all turns had valid tags
#   'efficiency_bonus': 0.70,      # closed on turn 3 of 10
#   'anchoring_reward': 0.60,      # opened near ideal anchor
#   'negotiation_progress_reward': 0.50   # controlled concessions
# }

score = score_trajectory(trajectory)   # e.g. 0.772
```

---

## ⚙️ Configuration

All behaviour is controlled via environment variables — no code changes needed.

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_TOKEN` | API key for LLM calls (primary alias for HF deployments) | **required** |
| `API_KEY` | API key override (takes precedence over `HF_TOKEN`) | — |
| `API_BASE_URL` | OpenAI-compatible inference endpoint | `https://router.huggingface.co/v1` |
| `SELLER_MODEL` | Model used to generate seller replies (server-side) | `Qwen/Qwen2.5-72B-Instruct` |
| `BUYER_MODEL` / `MODEL_NAME` | Model used by `inference.py` for buyer turns | `Qwen/Qwen2.5-72B-Instruct` |
| `ENV_BASE_URL` | Server URL used by `inference.py` | HF Space URL |
| `BUYER_TEMPERATURE` | Sampling temperature for buyer LLM | `0.7` |
| `SUCCESS_SCORE_THRESHOLD` | Minimum `score_trajectory` to count as success | `0.4` |
| `DEBUG` | Set to `1` / `true` to enable verbose debug logs | `false` |

**Minimal setup (HF token only):**

```bash
export HF_TOKEN=hf_...
export API_BASE_URL="https://router.huggingface.co/v1"
```

Both buyer and seller default to `Qwen/Qwen2.5-72B-Instruct` via the HF inference router. No OpenAI key required.

---

## ▶️ Running `inference.py`

`inference.py` runs a full evaluation loop — one episode per difficulty (`easy` → `medium` → `hard`) — against a live server and prints structured benchmark logs.

### Setup

```bash
export HF_TOKEN=hf_...
export API_BASE_URL="https://router.huggingface.co/v1"
export BUYER_MODEL="Qwen/Qwen2.5-72B-Instruct"                          # optional
export ENV_BASE_URL="https://viditostwal-price-negotiation.hf.space"    # or http://localhost:8000
```

### Run

```bash
# Run all three difficulties (easy → medium → hard)
uv run python inference.py

# Run a single difficulty
uv run python inference.py --difficulty easy
uv run python inference.py -d hard
```

### stdout format

The script emits exactly three line types per episode, suitable for benchmark harnesses:

```
[START] task=easy env=price_negotiation model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=I am rea reward=0.45 done=false error=null
[STEP]  step=2 action=That see reward=0.61 done=true  error=null
[END]   task=easy success=true steps=2 score=0.612 rewards=[0.45,0.61]
```

At the end of each episode, the full `reward_breakdown` dict is printed showing all six component scores.

---

## 📁 Project Structure

```
.
├── __init__.py               # Package exports
├── client.py                 # PriceNegotiationEnv (typed WebSocket client)
├── models.py                 # Action / Observation / State Pydantic models
├── reward.py                 # All 6 reward components + score_trajectory()
├── rollout.py                # Synchronous rollout runner
├── trajectory_types.py       # TrajectoryStep / TrajectoryResult dataclasses
├── inference.py              # Async evaluation script (benchmark entry point)
├── Dockerfile                # Multi-stage Docker build (openenv-base)
├── openenv.yaml              # OpenEnv Space deployment config
├── pyproject.toml            # Package metadata and dependencies
└── server/
    ├── app.py                # FastAPI app (create_app factory)
    ├── price_negotiation_environment.py  # Core environment logic
    ├── helper_functions.py   # OpenAI-compatible LLM client wrapper
    ├── dataset.json          # 3 negotiation scenarios (easy / medium / hard)
    └── requirements.txt      # Server-only pip dependencies
```
