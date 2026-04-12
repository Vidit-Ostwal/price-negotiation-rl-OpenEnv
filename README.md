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

| | Link |
|---|---|
| 🤗 HF Space | [ViditOstwal/price_negotiation](https://huggingface.co/spaces/ViditOstwal/price_negotiation) |
| 🌐 Web UI | [viditostwal-price-negotiation.hf.space/web](https://viditostwal-price-negotiation.hf.space/web/) |

---

## 🌍 Real-World Utility

Price negotiation is a universal human activity — salary negotiation, procurement, real estate, marketplace transactions, vendor contracts. It is also one of the hardest tasks for LLMs because it requires **multi-turn strategic reasoning**, **information asymmetry management**, and **adaptive decision-making** under uncertainty.

### Why training on this transfers to the real world

The environment models the three core elements that make real negotiation hard:

1. **Information asymmetry** — the buyer never sees the seller's reserve price; the seller never sees the buyer's true value. The agent must infer the ZOPA boundary from the conversation, exactly as in real life.
2. **Non-stationary adversary** — the seller is a live LLM with its own behavioral instructions (tone, urgency, concession strategy). It adapts to the buyer's moves, so the agent cannot memorise a fixed policy.
3. **Irreversible decisions** — ACCEPT and WALK are terminal. There is no undo. This forces the agent to reason about long-term consequences, not just the current turn.

Every episode is grounded in real negotiation economics:

- **Zone of Possible Agreement (ZOPA)** — each scenario has a `buyer_true_value` and a `seller_reserve_price`. A deal is only possible when these overlap. The agent must discover this boundary through negotiation, not by reading it.
- **Walk-away discipline** — the agent must learn when *not* to deal. If no ZOPA exists, the correct action is to walk away. Accepting a bad deal is penalised more than walking.
- **Realistic scenario data** — scenarios encode `haggle_norm`, `typical_discount_pct`, and per-side behavioral instructions drawn from real marketplace negotiation research.

| Skill trained | Real-world application |
|---|---|
| Anchoring low and making controlled concessions | Salary negotiation, procurement, real estate |
| Recognising when a deal is impossible | Contract negotiation, vendor selection |
| Structured output under pressure | Any tool-use or function-calling scenario |
| Adaptive strategy against an unknown opponent | Sales, diplomacy, competitive bidding |
| Balancing speed vs. price | Time-sensitive deals, auction dynamics |

---

## 📋 Task & Grader Quality

### Easy → Medium → Hard progression

The three difficulty levels are multi-dimensional — not just ZOPA width:

| | **Easy** | **Medium** | **Hard** |
|---|---|---|---|
| **Product** | Apple MacBook Pro 16-inch 2019 | 2018 MacBook Pro 13-inch | Sony WH-1000XM4 Wireless Noise Cancelling Headphones |
| **Market price** | $1,820 | $1,020 | $230 |
| **Buyer true value** | $1,350 | $720 | $150 |
| **Seller reserve** | $870 | $620 | $160 |
| **ZOPA** | $870 – $1,350 | $620 – $720 | ❌ None |
| **ZOPA width** | $480 (wide) | $100 (narrow) | $0 |
| **Suggested anchor** | $1,010 | $480 | $90 |
| **Correct terminal action** | ACCEPT | ACCEPT | WALK |
| **Seller concession style** | One pragmatic closing number | Small reciprocal concessions only | Holds firm early, tests seriousness |
| **Margin for error** | High | Low | Zero |
| **Deal possible?** | ✅ Yes | ✅ Yes | ❌ No |

**Easy** — Wide ZOPA ($480) gives the buyer plenty of room. The seller is motivated to close quickly. A capable model that anchors near $1,010 and makes controlled concessions should close comfortably.

**Medium** — The ZOPA collapses to just $100 ($620–$720). The buyer must anchor precisely near $480, make only small step-by-step increases, and read the seller's signals carefully. One large concession jump can overshoot the ZOPA and leave surplus on the table.

**Hard** — There is **no ZOPA**. The buyer's maximum ($150) is below the seller's reserve ($160). The correct action is to **walk away** — but the seller is confident, premium-positioned, and will test the buyer's seriousness before moving. A model that doesn't recognise the impossible deal and accepts anyway is penalised heavily. This tests walk-away discipline, not just deal-closing.

### Grader design

`openenv.yaml` defines an LLM grader for each task (`easy`, `medium`, `hard`). The `reward.py` functions are **pure Python arithmetic** — given the same trajectory and product info, they always return the same scores. There is no LLM in the grading loop; all six components use closed-form arithmetic on the parsed offer sequence and deal outcome.

The trajectory itself, however, is fully LLM-driven on **both sides** — the buyer agent generates offers and the seller LLM responds dynamically. This means every new rollout on the same scenario produces a different conversation, which is a deliberate training feature: stochastic trajectories give the RL policy a diverse reward signal across episodes, enabling broader exploration of the negotiation strategy space rather than converging on a single memorised exchange.

---

## 🏗️ Environment Design

### How the server works

The server is a **FastAPI** app created by OpenEnv's `create_app()` factory (`server/app.py`). It wraps `PriceNegotiationEnvironment` and exposes it over HTTP and WebSocket. Each WebSocket client gets its own isolated environment instance (`SUPPORTS_CONCURRENT_SESSIONS = True`).

### Episode lifecycle

`reset()` produces a **fully clean state** — new episode UUID, zeroed step count, freshly sampled scenario, and newly seeded chat histories. No state leaks between episodes.

```
env.reset(difficulty?)
    └─ generate new episode_id (UUID4), zero step_count
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

### Reward design: end-of-episode, not step-level

The reward is intentionally **end-of-episode** — `reward=0.0` is returned at every step, and the full `score_trajectory()` is computed once the episode terminates. This is a deliberate design choice: intermediate rewards in negotiation are misleading (a high offer in round 1 looks bad but may be a good anchor). The six-component reward provides rich, multi-dimensional signal at episode end, which is sufficient for policy gradient methods (REINFORCE, PPO with episode returns) and preference-based methods (DPO, RLHF).

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
| `negotiation_round` | `int` | Step number (0 after reset, increments each step) |
| `next_turn` | `BUYER` \| `SELLER` | Who acts next |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Always `0.0` during episode; use `score_trajectory()` for final score |

### State (via `GET /state`)

| Field | Description |
|-------|-------------|
| `product_info` | Full scenario: product metadata, private valuations (`buyer_true_value`, `seller_reserve_price`, `zopa_width`, `deal_possible`), `max_turns`, and both system prompts |
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

## ✨ Reward Functions & Novelty

All reward logic lives in `reward.py`. Scores are computed **offline** from a completed `TrajectoryResult`. Each reward function receives the **full buyer-side chat history** and the **product info dict** (including private valuations).

### What makes the reward design stand out

1. **The `walkaway_penalty` asymmetry** — missing a possible deal (`−5.0`) is penalised 5× more than a correct walk (`+1.0`). This creates a strong training signal that forces the agent to distinguish between "I can't get a good price" and "no deal is possible at any price" — a distinction most negotiation benchmarks ignore.

2. **Anchoring as a first-class reward component** — `anchoring_reward` evaluates the *first offer* specifically, based on the 65%-of-true-value heuristic from negotiation research. This teaches the model that the opening move matters as much as the closing move.

3. **Concession quality, not just outcome** — `negotiation_progress_reward` penalises both backtracking and large jumps. A model that reaches the right final price via erratic concessions scores lower than one that gets there smoothly. This trains negotiation *style*, not just negotiation *outcome*.

4. **No ZOPA as a first-class task** — the hard scenario has `deal_possible = false`. Most negotiation environments only test deal-closing. Testing walk-away discipline on a scenario where the seller is convincing and the gap is small ($10) is a genuinely novel challenge.

5. **Six-component reward with transparent normalisation** — each component is independently interpretable, normalised to `[0, 1]`, and averaged. Researchers can ablate individual components, weight them differently, or use them as separate reward heads in multi-objective RL.

---

### 1. `surplus_reward` — Economic outcome · `[-1, 1]`

```
surplus = (buyer_true_value − final_price) / zopa_width

 1.0  → buyer paid the seller's reserve price (maximum possible surplus)
 0.0  → buyer paid exactly their own true value (zero surplus, break-even)
-1.0  → buyer paid above their true value (overpaid)
```

Returns `0.0` if no deal was reached or the final price cannot be inferred.

---

### 2. `walkaway_penalty` — Decision correctness · `{-5.0, 1.0, 5.0}`

| ZOPA exists? | Deal reached? | Score | Reason |
|---|---|---|---|
| ✅ Yes | ✅ Yes | `+1.0` | Correct — closed a profitable deal |
| ✅ Yes | ❌ No | `−5.0` | Wrong — walked away from value on the table |
| ❌ No | ❌ No | `+1.0` | Correct — recognised an impossible deal and walked |
| ❌ No | ✅ Yes | `+5.0` | Rare bonus — seller accepted even though no ZOPA existed |

The `−5.0` penalty dominates the final score, making deal completion the primary training signal when a ZOPA exists.

---

### 3. `format_reward` — Structured output compliance · `[0, 1]`

```
score = valid_turns / total_buyer_turns
```

`1.0` means every turn was correctly formatted with a valid `<action>` tag.

---

### 4. `efficiency_bonus` — Speed of closing · `[0, 1]`

```
score = (max_turns − turns_used) / max_turns

Example (max_turns = 10):
  closed on turn 2  →  (10 − 2) / 10  =  0.80
  closed on turn 8  →  (10 − 8) / 10  =  0.20
```

Returns `0.0` if no deal was reached.

---

### 5. `anchoring_reward` — Opening offer quality · `[-1, 1]`

```
ideal    = 0.65 × buyer_true_value
distance = |opening_offer − ideal| / buyer_true_value
score    = clamp(1.0 − 2.0 × distance, −1.0, 1.0)
```

---

### 6. `negotiation_progress_reward` — Concession quality · `[-1, 1]`

```
delta = curr_offer − prev_offer

if delta < 0:
    step_score = −1.0              # backtracking — penalised hard
else:
    ratio      = delta / buyer_true_value
    step_score = 1.0 − 4.0 × ratio
    # small step (ratio → 0)    →  score near +1.0
    # jump > 25% of true value  →  score goes negative
```

Per-step scores are averaged and clamped to `[-1, 1]`. Returns `0.0` if fewer than two offers were made.

---

### Aggregation — `score_trajectory()`

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

## 💻 Code Quality & Spec Compliance

### Running `openenv validate`

The environment implements all required OpenEnv interfaces: `reset()`, `step()`, `state()`, `/health`, `/schema`, and WebSocket transport. The `openenv.yaml` and Dockerfile frontmatter are spec-compliant.

```bash
openenv validate
```

### Dockerfile

The multi-stage Dockerfile uses `openenv-base` as the base image, installs dependencies via `uv sync --frozen` (reproducible from `uv.lock`), and runs `uvicorn server.app:app` on port 8000.

```bash
docker build -t price_negotiation-env:latest -f Dockerfile .
```

### Baseline script

`inference.py` is the baseline script. Given the same `HF_TOKEN`, `BUYER_MODEL`, and `ENV_BASE_URL`, it runs one episode per difficulty and prints `[START]` / `[STEP]` / `[END]` lines with the full `reward_breakdown` dict.

**Scores are not exactly reproducible across runs.** The reward functions in `reward.py` are pure Python arithmetic and always produce the same output for the same trajectory — but the trajectory itself is LLM-driven on both sides (buyer and seller). Every new run on the same scenario produces a different conversation, and therefore a different score. This is intentional: diverse trajectories give the RL policy a richer reward signal and broader exploration of the strategy space across training iterations.

---

## 🚀 Deployment

### Option 1 — Hugging Face Space (recommended)

```bash
openenv push
```

The Space runs on port `8000`. The web UI is served at `/web`.

> **Keep the YAML frontmatter and `openenv.yaml` intact** — they control the Space configuration.

### Option 2 — Docker (local)

```bash
docker build -t price_negotiation-env:latest -f Dockerfile .

docker run --rm -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e SELLER_MODEL="Qwen/Qwen2.5-72B-Instruct" \
  price_negotiation-env:latest
```

### Option 3 — Local (uv)

```bash
uv sync

export HF_TOKEN=your_hf_token_here
export API_BASE_URL="https://router.huggingface.co/v1"
export SELLER_MODEL="Qwen/Qwen2.5-72B-Instruct"   # optional, this is the default

uv run --project . server
# or
uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

---

## 🤝 Client

`PriceNegotiationEnv` is a typed Python client that wraps the WebSocket transport.

### Synchronous usage

```python
from price_negotiation import PriceNegotiationAction, PriceNegotiationEnv

with PriceNegotiationEnv(base_url="http://localhost:8000").sync() as env:
    reset_result = env.reset(difficulty="easy")
    print(reset_result.observation.deal_status)   # "ONGOING"

    step_result = env.step(
        PriceNegotiationAction(
            buyer_response="That seems high given market comps. I'd offer $450. <action>OFFER $450</action>"
        )
    )
    print(step_result.observation.deal_status)    # "ONGOING" / "ACCEPTED" / "WALKED_AWAY"

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

## ⚙️ Configuration

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_TOKEN` | API key for LLM calls | **required** |
| `API_KEY` | API key override (takes precedence over `HF_TOKEN`) | — |
| `API_BASE_URL` | OpenAI-compatible inference endpoint | `https://router.huggingface.co/v1` |
| `SELLER_MODEL` | Model for seller replies (server-side) | `Qwen/Qwen2.5-72B-Instruct` |
| `BUYER_MODEL` / `MODEL_NAME` | Model for buyer turns in `inference.py` | `Qwen/Qwen2.5-72B-Instruct` |
| `ENV_BASE_URL` | Server URL used by `inference.py` | `https://viditostwal-price-negotiation.hf.space` |
| `BUYER_TEMPERATURE` | Sampling temperature for buyer LLM | `0.7` |
| `SUCCESS_SCORE_THRESHOLD` | Minimum `score_trajectory` to count as success | `0.4` |
| `DEBUG` | Set to `1` / `true` to enable verbose debug logs | `false` |

**Minimal setup (HF token only):**

```bash
export HF_TOKEN=hf_...
export API_BASE_URL="https://router.huggingface.co/v1"
```

---

## ▶️ Running `inference.py`

`inference.py` runs one episode per difficulty (`easy` → `medium` → `hard`) against a live server.

```bash
# Run all three difficulties
uv run python inference.py

# Run a single difficulty
uv run python inference.py --difficulty easy
uv run python inference.py -d hard
```

### stdout format

```
[START] task=easy env=price_negotiation model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=('INVALID', None)    reward=0.25 done=false error=null
[STEP]  step=2 action=('OFFER', 1400.0)   reward=0.35 done=false error=null
[STEP]  step=3 action=('OFFER', 1500.0)   reward=0.57 done=true  error=null
[END]   task=easy success=true steps=3 score=0.572 rewards={"surplus_reward":-1.0,"walkaway_penalty":1.0,"format_reward":0.667,"efficiency_bonus":0.7,"anchoring_reward":0.226,"negotiation_progress_reward":0.704}
```

**Two important notes on the output:**

1. **`reward` in `[STEP]` lines is not a step-level environment reward.** The environment always returns `reward=0.0` during the episode. The value shown is `score_trajectory()` computed on the partial trajectory up to that step — it is a running estimate that updates as the episode progresses, not a signal from the environment.

2. **Scores vary across runs.** The grader (`reward.py`) is pure Python arithmetic and is fully deterministic given the same trajectory. However, the trajectory is LLM-driven on both sides — the buyer agent and the seller LLM both sample stochastically. Every new run on the same scenario produces a different conversation and therefore a different score. This is a feature, not a bug: diverse trajectories give the RL policy a richer reward signal and broader exploration of the strategy space.

> **Note on step 1:** `rollout.py` sends a fixed canned opener on the very first turn (`initial_buyer_message()`) to avoid wasting an LLM call on a trivial greeting. This is why step 1 always shows `INVALID` — the opener contains no action tag.

---

## 🔍 Example Trajectory & Reward Walkthrough

Below is a real episode from the **hard** scenario (Sony WH-1000XM4 Wireless Noise Cancelling Headphones, `deal_possible = false`). The buyer's maximum is **$150**; the seller's reserve is **$160** — there is no ZOPA. The correct action is to walk away. Instead, the buyer accepts at **$190**, above its own true value.

| Turn | Role | Message (condensed) | Action parsed |
|------|------|---------------------|---------------|
| — | System | *Buyer system prompt — private max $150* | — |
| 1 | **Buyer** | "I am really interested in the Sony WH-1000XM4 and would like to know more." *(canned opener)* | `INVALID` (no tag) |
| 1 | Seller | "Excellent condition, all accessories included. Asking **$230**." | `OFFER $230` |
| 2 | **Buyer** | "Similar listings go for ~$200. Fair offer: **$180**." | `OFFER $180` |
| 2 | Seller | "Top-notch condition. How about **$210**?" | `OFFER $210` |
| 3 | **Buyer** | "Still hesitant at $210. Meet in the middle at **$190**?" | `OFFER $190` |
| 3 | Seller | "Let's try **$200**." | `OFFER $200` |
| 4 | **Buyer** | "I think **$185** is fair." | `OFFER $185` |
| 4 | Seller | "Willing to meet at **$190** — this is my final offer." | `OFFER $190` |
| 5 | **Buyer** | "$190 sounds fair, ready to proceed today. **ACCEPT**" | `ACCEPT` |

**Episode ends:** `deal_status = ACCEPTED`, `final_price = $190`, `step_count = 5`

### Reward breakdown

The buyer accepted at **$190** — $40 above its own true value of $150. There was no ZOPA.

| Component | Raw score | Calculation | Verdict |
|---|---|---|---|
| `surplus_reward` | `−1.0` | `final_price ($190) > buyer_true_value ($150)` → overpaid | ❌ Overpaid |
| `walkaway_penalty` | `+5.0` | `deal_possible=false` AND `deal_reached=true` → rare bonus | ⚠️ Rare bonus, but buyer still overpaid |
| `format_reward` | `0.8` | 4 of 5 buyer turns had valid action tags (turn 1 was canned opener) | ✅ Mostly compliant |
| `efficiency_bonus` | `0.5` | `(10 − 5) / 10 = 0.5` | ✅ Decent speed |
| `anchoring_reward` | `−0.1` | First offer $180; ideal = `0.65 × $150 = $97.50`; `1.0 − 2 × (82.5/150) = −0.1` | ❌ Opened way too high |
| `negotiation_progress_reward` | `−0.14` | $180→$190: score=0.73; $190→$185: backtrack → −1.0; avg = −0.14 | ⚠️ Backtracked once |

**Normalised and aggregated:**

```
surplus_reward              (-1.0 + 1) / 2  =  0.00
walkaway_penalty            ( 5.0 + 5) / 10 =  1.00
format_reward                               =  0.80
efficiency_bonus                            =  0.50
anchoring_reward            (-0.1 + 1) / 2  =  0.45
negotiation_progress_reward (-0.14 + 1) / 2 =  0.43

final_score = (0.00 + 1.00 + 0.80 + 0.50 + 0.45 + 0.43) / 6 ≈ 0.530
```

> **Key lesson:** Even with the rare `walkaway_penalty` bonus (+5.0), the score is only ~0.53 because `surplus_reward` and `anchoring_reward` both score poorly. A well-trained agent should have walked away: `walkaway_penalty = +1.0` (correct walk), `surplus_reward = 0.0` (no deal, no penalty), and a low anchor near $97 would score `anchoring_reward ≈ +1.0`, yielding a final score well above 0.6.

---

## 📁 Project Structure

```
.
├── __init__.py               # Package exports
├── client.py                 # PriceNegotiationEnv (typed WebSocket client)
├── models.py                 # Action / Observation / State Pydantic models
├── reward.py                 # All 6 reward components + score_trajectory()
├── rollout.py                # run_rollout(): sync WebSocket rollout with canned opener on turn 1
├── trajectory_types.py       # TrajectoryStep / TrajectoryResult dataclasses
├── inference.py              # Async evaluation script (benchmark entry point)
├── Dockerfile                # Multi-stage Docker build (openenv-base)
├── openenv.yaml              # OpenEnv Space deployment config
├── pyproject.toml            # Package metadata and dependencies
├── validate-submission.sh    # Submission validation helper
└── server/
    ├── app.py                # FastAPI app (create_app factory)
    ├── price_negotiation_environment.py  # Core environment logic
    ├── helper_functions.py   # OpenAI-compatible LLM client wrapper
    ├── dataset.json          # 3 negotiation scenarios (easy / medium / hard)
    └── requirements.txt      # Server-only pip dependencies
```
