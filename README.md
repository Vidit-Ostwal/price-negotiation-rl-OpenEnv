---
title: Price Negotiation Environment Server
emoji: 🎭
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# 🎭 Price Negotiation Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) RL environment where an LLM agent plays the **buyer** and negotiates against an LLM-powered **seller** over real marketplace listings. The server exposes standard OpenEnv HTTP and WebSocket endpoints and is pre-configured for Hugging Face Spaces deployment.

| | Link |
|---|---|
| 🤗 HF Space | [ViditOstwal/price_negotiation](https://huggingface.co/spaces/ViditOstwal/price_negotiation) |
| 🌐 Web UI | [viditostwal-price-negotiation.hf.space](https://viditostwal-price-negotiation.hf.space) |
| 🔗 GitHub | [Vidit-Ostwal/price-negotiation-rl-OpenEnv](https://github.com/Vidit-Ostwal/price-negotiation-rl-OpenEnv) |

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

When **`ENABLE_WEB_INTERFACE`** is set (see [Configuration](#configuration)), the app **replaces** the default `POST /reset`, `POST /step`, and `GET /state` handlers with a **single persistent browser session** backed by a dedicated `PriceNegotiationEnvironment` instance. That mode serves the static UI from `server/static/` at **`/`** (the Docker image and `run_local.sh` enable this by default). It also builds a `TrajectoryResult` after each step and attaches **running** `score_trajectory()` values plus per-component breakdowns to JSON responses (see [Observation space](#observation-space)). The WebSocket API and other OpenEnv routes are unchanged.

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

### Reward design: end-of-episode core, running score in the web UI

In the **`PriceNegotiationEnvironment`** implementation, **`observation.reward` stays `0.0` at every step** so the canonical OpenEnv contract is unchanged: grading is driven by **`reward_breakdown()` + `score_trajectory()` on a finished `TrajectoryResult`**.

When the **bundled HTTP UI** is enabled (`ENABLE_WEB_INTERFACE=true`), each **`POST /step`** still returns that canonical observation shape, but the response also fills **`reward`** (and mirrored fields under **`observation`**) with **`score_trajectory()` evaluated on the trajectory so far, plus **`reward_breakdown`**, **`reward_weights`**, and **`reward_components`**. Those extras power the Reward card’s **View breakdown** modal (per-component **`2 × 2` metric tiles** plus summary strip, optional GitHub shortcut in the page header). They do **not** change how the seller or environment logic behaves; they are telemetry for debugging and demos.

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
| `reward` | `float` | `0.0` from the core environment during the episode; with **`ENABLE_WEB_INTERFACE`**, HTTP `/step` overwrites serialization with running `score_trajectory()` |
| `reward_breakdown` | `dict[str, float]` \| `null` | Raw per-component scores; set when the web overlay computes breakdowns (`null` on reset path until first graded step / standard clients leave unset) |
| `reward_weights` | `dict[str, float]` \| `null` | Fixed **⅙** weight per component matching `score_trajectory()`’s equal mean |
| `reward_components` | `dict[str, object]` \| `null` | Per-component `{ raw, score, weight, weighted_score }` where `score` applies the aggregation map (below) before averaging |

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

With **`ENABLE_WEB_INTERFACE=true`**, `POST /reset`, `POST /step`, and `GET /state` use the web session described above (`GET /state` returns `PriceNegotiationState` only — no synthesized reward metadata). Static assets live under **`/static`**; the HTML shell links a small **favicon** and stylesheet/script cache-busting query strings.

---

## ✨ Reward Functions & Novelty

All reward logic lives in `reward.py`. Scores are computed **offline** from a completed `TrajectoryResult`. Each reward function receives the **full buyer-side chat history** and the **product info dict** (including private valuations).

### What makes the reward design stand out

1. **The `walkaway_penalty` asymmetry** — missing a possible deal (`−5.0`) is penalised 5× more than a correct walk (`+1.0`). This creates a strong training signal that forces the agent to distinguish between "I can't get a good price" and "no deal is possible at any price" — a distinction most negotiation benchmarks ignore.

2. **Anchoring as a first-class reward component** — `anchoring_reward` evaluates the *first offer* specifically, based on the 65%-of-true-value heuristic from negotiation research. This teaches the model that the opening move matters as much as the closing move.

3. **Concession quality, not just outcome** — `negotiation_progress_reward` penalises both backtracking and large jumps. A model that reaches the right final price via erratic concessions scores lower than one that gets there smoothly. This trains negotiation *style*, not just negotiation *outcome*.

4. **No ZOPA as a first-class task** — the hard scenario has `deal_possible = false`. Most negotiation environments only test deal-closing. Testing walk-away discipline on a scenario where the seller is convincing and the gap is small ($10) is a genuinely novel challenge.

5. **Six-component reward with explicit aggregation scale** — each component stays on its interpretable native scale (`reward_breakdown`); `score_trajectory()` averages them on a **`[-1, 1]`-compatible map** (`walkaway_penalty` contributes `raw / 5`, everything else is already in **`[-1, 1]`** or **`[0, 1]`** as documented per function). Researchers can still ablate, reweight outside the helper, or use components as separate reward heads.

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

### 2. `walkaway_penalty` — Decision correctness · `{0.0, −1.0, −5.0, +1.0, +5.0}`

`reward_state()` now carries **`deal_status`** (episode still **ONGOING**, or terminal status) and **`turn`** (`negotiation_round` at the terminal observation; `0` if no buyer step landed yet).

| Situation | Score | Reason |
|-----------|-------|--------|
| `deal_status == ONGOING` | `0.0` | Negotiation unfinished — neither reward nor penalise the terminal rule yet |
| Deal reached **on negotiation round `1`** | `−1.0` | **Premature acceptance** — the buyer closed before probing the seller |
| ZOPA exists & deal reached (not caught above) | `+1.0` | Economically sensible close |
| ZOPA exists & walked away | `−5.0` | Wrong — surplus left on the table |
| No ZOPA & walked away | `+1.0` | Correct restraint |
| No ZOPA & deal reached | `+5.0` | Rare path — seller conceded into a deal that looked impossible on paper |

The `−5.0` penalty still dominates when a ZOPA existed and the buyer gave up.

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

Returns `0.0` if no deal was reached. **First-round closes** also return `0.0` (they are treated as premature for efficiency, consistent with `walkaway_penalty`).

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

Component values are averaged on an aggregate scale suitable for **`[-1, 1]`** summaries:

```
surplus_reward              → unchanged  (already [−1, 1])
walkaway_penalty            → raw / 5    (maps {−5, …, +5} into [−1, 1])
format_reward               → unchanged  ([0, 1])
efficiency_bonus            → unchanged  ([0, 1])
anchoring_reward            → unchanged  ([−1, 1])
negotiation_progress_reward → unchanged  ([−1, 1])

final_score = mean of all six  ∈ [−1, 1]
```

```python
from price_negotiation.reward import reward_breakdown, score_trajectory

breakdown = reward_breakdown(trajectory)
# {
#   'surplus_reward': 0.83,        # captured 83% of ZOPA
#   'walkaway_penalty': 1.0,       # raw +1.0 → +0.2 on aggregate scale (+1÷5 before the six-way mean)
#   'format_reward': 1.0,          # all turns had valid tags
#   'efficiency_bonus': 0.70,      # closed on turn 3 of 10
#   'anchoring_reward': 0.60,      # opened near ideal anchor
#   'negotiation_progress_reward': 0.50   # controlled concessions
# }

score = score_trajectory(trajectory)   # e.g. ≈ +0.64 for the illustrative numbers above
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

The Space runs on port `8000`. The Docker image enables **`ENABLE_WEB_INTERFACE`**, so the buyer playground is served at **`/`** (static files under **`/static`**).

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

# Convenience script: launches uvicorn with ENABLE_WEB_INTERFACE=true
./run_local.sh

# Or run the API without swapping HTTP routes / without the bundled UI shell:
uv run --project . server
# or
ENABLE_WEB_INTERFACE=false uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
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
| `SUCCESS_SCORE_THRESHOLD` | Minimum `score_trajectory` to count as success (same **[-1, 1]** aggregate as grading) | `0.4` |
| `ENABLE_WEB_INTERFACE` | Replace HTTP `/reset` `/step` `/state`, serve `server/static` at **`/`** | `false` locally unless set; **`true`** in Docker / `run_local.sh` |
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
```

### stdout format

Each `[STEP]` line prints **`reward=`** as the cumulative `score_trajectory()` aggregate on the **prefix trajectory opened so far** (same **`[-1, 1]`** mean as the final line). The environment’s internal `observation.reward` remains `0.0` until you grade offline; this column is added by the logger for visibility.

```
[START] task=easy env=price_negotiation model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=('INVALID', None)    reward=… done=false error=null
[STEP]  step=2 action=('OFFER', 1400.0)   reward=… done=false error=null
[STEP]  step=3 action=('OFFER', 1500.0)   reward=… done=true  error=null
[END]   task=easy success=false steps=3 score=0.249 rewards={"surplus_reward":-1.0,"walkaway_penalty":1.0,"format_reward":0.667,"efficiency_bonus":0.7,"anchoring_reward":0.226,"negotiation_progress_reward":0.704}
```

Using the printed **`rewards=`** breakdown with the aggregation map above,

`score = (−1.0 + 1.0/5 + 0.667 + 0.7 + 0.226 + 0.704) / 6 ≈ 0.25`, below the default **`SUCCESS_SCORE_THRESHOLD=0.4`**, hence `success=false` in this toy tail.

**Two important notes on the output:**

1. **`reward` in `[STEP]` lines mirrors `score_trajectory()` on the prefix trajectory.** Core `Observation.reward` stays `0.0` until you grade offline, but the logger reads the cumulative breakdown and reports the same aggregate you would get from `score_trajectory()` at that prefix — now on the **`[-1, 1]`** mean scale.

2. **Scores vary across runs.** The grader (`reward.py`) is pure Python arithmetic and is fully deterministic given the same trajectory. However, the trajectory is LLM-driven on both sides — the buyer agent and the seller LLM both sample stochastically. Every new run on the same scenario produces a different conversation and therefore a different score. This is a feature, not a bug: diverse trajectories give the RL policy a richer reward signal and broader exploration of the strategy space.

Tuning note: **`SUCCESS_SCORE_THRESHOLD` defaults to `0.4` on the new scale.** If your evaluation harness was calibrated against the retired `[0, 1]` normalisation bump that threshold upward or downward after spot-checking a few traces.

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

**Aggregated (current `score_trajectory()` — mean on native / scaled components):**

```
surplus_reward              =  −1.00      (already on [−1, 1])
walkaway_penalty            =    5.0 / 5  =  +1.00
format_reward               =   +0.80
efficiency_bonus            =   +0.50
anchoring_reward            =   −0.10      (already on [−1, 1])
negotiation_progress_reward =   −0.14      (already on [−1, 1])

final_score = (−1.00 + 1.00 + 0.80 + 0.50 − 0.10 − 0.14) / 6 ≈ 0.18
```

> **Key lesson:** The rare `walkaway_penalty` **+5.0** raw bonus still maps to **+1.0** on the aggregate channel (`÷5`), but **negative `surplus_reward` pulls the mean down hard** instead of being squashed toward the centre of `[0, 1]` like under the retired normalisation. A model that anchors low and walks on impossible deals climbs the scale much faster than one that trophies a bad surplus outcome.

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
├── run_local.sh              # Start uvicorn with ENABLE_WEB_INTERFACE=true (browser UI at /)
└── server/
    ├── app.py                # FastAPI app (create_app factory + optional web session layer)
    ├── price_negotiation_environment.py  # Core environment logic
    ├── helper_functions.py   # OpenAI-compatible LLM client wrapper
    ├── dataset.json          # 3 negotiation scenarios (easy / medium / hard)
    ├── requirements.txt      # Server-only pip dependencies
    └── static/
        ├── index.html        # Buyer UI shell (reward + product modals, quick actions)
        ├── app.js            # Calls /reset · /step · /state against the web session
        ├── styles.css        # Layout / reward breakdown grid styling
        └── favicon.svg       # Lightweight tab icon for the playground
```
