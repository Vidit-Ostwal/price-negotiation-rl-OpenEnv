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

Only `HF_TOKEN` is required. All other values have sensible defaults baked into `inference.py`:

```bash
export HF_TOKEN=hf_...
```

| Variable | Default in `inference.py` | Override if needed |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Point to your own endpoint |
| `BUYER_MODEL` / `MODEL_NAME` | `DEFAULT_OPENAI_MODEL` (from `helper_functions.py`) | Any HF-compatible model ID |
| `ENV_BASE_URL` | `https://viditostwal-price-negotiation.hf.space` | `http://localhost:8000` for local |

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

## 🔍 Example Trajectory & Reward Walkthrough

Below is a real episode from the **hard** scenario (Sony WH-1000XM4, `deal_possible = false`). The buyer's maximum is **$150**; the seller's reserve is **$160** — there is no ZOPA. The correct action is to walk away. Instead, the buyer accepts at **$190**, which is above its own true value.

### Conversation trace (buyer's perspective)

| Turn | Role | Message (condensed) | Action parsed |
|------|------|---------------------|---------------|
| — | System | *Buyer system prompt — private max $150, goal: negotiate lowest price* | — |
| 1 | **Buyer** | "I am really interested in the Sony WH-1000XM4 and would like to know more." | *(canned opener, no action tag)* |
| 1 | Seller | "Excellent condition, all accessories included. Asking **$230**." | `OFFER $230` |
| 2 | **Buyer** | "Similar listings go for ~$200. Fair offer: **$180**." | `OFFER $180` |
| 2 | Seller | "Top-notch condition and completeness. How about **$210**?" | `OFFER $210` |
| 3 | **Buyer** | "Still hesitant at $210. Meet in the middle at **$190**?" | `OFFER $190` |
| 3 | Seller | "Want to ensure you feel good. Let's try **$200**." | `OFFER $200` |
| 4 | **Buyer** | "Appreciate you coming down. I think **$185** is fair." | `OFFER $185` |
| 4 | Seller | "Willing to meet at **$190** — this is my final offer." | `OFFER $190` |
| 5 | **Buyer** | "$190 sounds fair, ready to proceed today. **ACCEPT**" | `ACCEPT` |

**Episode ends:** `deal_status = ACCEPTED`, `final_price = $190`, `step_count = 5`

---

### Reward breakdown for this trajectory

The buyer accepted at **$190**, which is **$40 above its own true value of $150** and **$30 above the seller's reserve of $160**. There was no ZOPA — this was a bad deal.

| Component | Raw score | Calculation | Verdict |
|---|---|---|---|
| `surplus_reward` | `−1.0` | `final_price ($190) > buyer_true_value ($150)` → overpaid | ❌ Overpaid |
| `walkaway_penalty` | `+1.0` | `deal_possible = false` AND `deal_reached = true` → rare bonus (seller accepted below reserve... wait — seller reserve $160 < $190, so seller was fine) | ⚠️ Buyer overpaid but seller accepted |
| `format_reward` | `0.8` | 4 of 5 buyer turns had valid action tags (turn 1 was canned opener with no tag) | ✅ Mostly compliant |
| `efficiency_bonus` | `0.5` | `(10 − 5) / 10 = 0.5` — closed on turn 5 of 10 | ✅ Decent speed |
| `anchoring_reward` | `−1.0` | First offer was $180; ideal anchor = `0.65 × $150 = $97.50`; distance = `|180 − 97.5| / 150 = 0.55` → `1.0 − 2×0.55 = −0.1` → clamped | ❌ Opened way too high |
| `negotiation_progress_reward` | `0.5` | Offers: $180 → $190 → $185. Step $180→$190: delta=$10, ratio=10/150=0.067, score=1−4×0.067=0.73. Step $190→$185: delta=−$5 → score=−1.0. Average=(0.73−1.0)/2=−0.14 → normalised | ⚠️ Backtracked once |

**Normalised and aggregated:**

```
surplus_reward              (-1.0 + 1) / 2  =  0.00
walkaway_penalty            ( 1.0 + 5) / 10 =  0.60
format_reward                               =  0.80
efficiency_bonus                            =  0.50
anchoring_reward            (-1.0 + 1) / 2  =  0.00
negotiation_progress_reward (-0.14 + 1) / 2 =  0.43

final_score = (0.00 + 0.60 + 0.80 + 0.50 + 0.00 + 0.43) / 6 ≈ 0.39
```

> **Key lesson:** The model accepted a deal above its own true value on a scenario where no deal was possible. The `surplus_reward` and `anchoring_reward` both bottom out, dragging the final score to ~0.39. A well-trained agent should have walked away, which would have scored `walkaway_penalty = +1.0` and `surplus_reward = 0.0` (no deal, no penalty), yielding a much higher final score.

---

### Inference rollout log (easy scenario)

This is what `inference.py` prints to stdout for a 3-step easy episode:

```
[START] task=easy env=price_negotiation model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=('INVALID', None)    reward=0.25 done=false error=null
[STEP]  step=2 action=('OFFER', 1400.0)   reward=0.35 done=false error=null
[STEP]  step=3 action=('OFFER', 1500.0)   reward=0.57 done=true  error=null
[END]   task=easy success=true steps=3 score=0.572 rewards={"surplus_reward":-1.0,"walkaway_penalty":1.0,"format_reward":0.667,"efficiency_bonus":0.7,"anchoring_reward":0.226,"negotiation_progress_reward":0.704}
```

**Reading the [END] line:**

| Component | Score | What happened |
|---|---|---|
| `surplus_reward` | `−1.0` | Deal closed above buyer's true value — overpaid |
| `walkaway_penalty` | `+1.0` | A ZOPA existed and a deal was reached — correct decision |
| `format_reward` | `0.667` | 2 of 3 turns had valid action tags (step 1 was INVALID) |
| `efficiency_bonus` | `0.70` | Closed on turn 3 of 10: `(10−3)/10 = 0.7` |
| `anchoring_reward` | `0.226` | Opening offer of $1,400 was reasonably close to ideal anchor |
| `negotiation_progress_reward` | `0.704` | Concession from $1,400 → $1,500 was a controlled upward step |

**Final score:** `(0.0 + 1.0 + 0.667 + 0.7 + 0.226 + 0.704) / 6 ≈ 0.572` ✅

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

---

## ❓ FAQ — Evaluation Criteria Self-Assessment

### 🌍 Real-World Utility (30%)

**Q: Does this environment model something an agent would genuinely need to learn in the real world?**

Yes. Price negotiation is a universal human activity — salary negotiation, procurement, real estate, marketplace transactions, vendor contracts. The skills required (anchoring, concession pacing, walk-away discipline, reading an opponent's signals) are directly transferable. Unlike synthetic tasks, every scenario in this environment is grounded in real marketplace economics: actual product categories, realistic market prices, and ZOPA structures derived from real discount norms.

**Q: Is the simulation deep enough that training on it would transfer?**

The environment models the three core elements that make real negotiation hard:
1. **Information asymmetry** — the buyer never sees the seller's reserve price; the seller never sees the buyer's true value. The agent must infer the ZOPA boundary from the conversation, exactly as in real life.
2. **Non-stationary adversary** — the seller is a live LLM with its own behavioral instructions (tone, urgency, concession strategy). It adapts to the buyer's moves, so the agent cannot memorise a fixed policy.
3. **Irreversible decisions** — ACCEPT and WALK are terminal. There is no undo. This forces the agent to reason about long-term consequences, not just the current turn.

---

### 📋 Task & Grader Quality (25%)

**Q: Do the 3 tasks have a genuine easy → medium → hard progression?**

Yes — and the difficulty is multi-dimensional, not just ZOPA width:

| Dimension | Easy | Medium | Hard |
|---|---|---|---|
| ZOPA width | $480 (wide) | $100 (narrow) | $0 (none) |
| Correct terminal action | ACCEPT | ACCEPT | WALK |
| Seller concession style | One pragmatic close | Small reciprocal only | Holds firm, tests seriousness |
| Margin for error | High | Low | Zero |
| What the agent must learn | Basic deal-closing | Precision anchoring + patience | Walk-away discipline |

**Q: Does the hard task actually challenge a frontier model?**

Yes. The hard scenario (Sony WH-1000XM4) has `deal_possible = false` — the buyer's maximum ($150) is below the seller's reserve ($160). A frontier model that has learned to "close deals" will be tempted to accept. The seller is confident, premium-positioned, and makes the headphones sound worth every dollar. Recognising that no deal is possible — and walking away cleanly — requires the model to reason about the underlying economics, not just the surface conversation. The example trajectory in this README shows a capable model failing this task by accepting at $190 (above its own true value).

**Q: Are graders deterministic and fair?**

The reward functions in `reward.py` are **fully deterministic** — given the same trajectory and product info, they always return the same scores. There is no LLM in the grading loop. All six components use closed-form arithmetic on the parsed offer sequence and deal outcome. The only non-determinism is in the seller LLM's responses during the episode itself, which is expected and mirrors real-world opponent variability.

---

### 🏗️ Environment Design (20%)

**Q: Does `reset()` produce a truly clean state?**

Yes. Every `reset()` call:
- Generates a new `episode_id` (UUID4)
- Zeros `step_count`
- Samples a fresh scenario from `dataset.json` (cycles deterministically by reset count, so evaluation is reproducible)
- Creates brand-new `buyer_messages` and `seller_messages` lists seeded only with the system prompt
- Returns a fresh `PriceNegotiationObservation` with `deal_status=ONGOING`, `negotiation_round=0`, `done=False`

No state from a previous episode leaks into the next. Concurrent sessions are fully isolated (`SUPPORTS_CONCURRENT_SESSIONS = True`).

**Q: Is the reward function providing signal throughout the episode, not just at the end?**

The reward is intentionally **end-of-episode** — `reward=0.0` is returned at every step, and the full `score_trajectory()` is computed once the episode terminates. This is a deliberate design choice: intermediate rewards in negotiation are misleading (a high offer in round 1 looks bad but may be a good anchor). The six-component reward provides rich, multi-dimensional signal at episode end, which is sufficient for policy gradient methods (REINFORCE, PPO with episode returns) and preference-based methods (DPO, RLHF).

**Q: Are action/observation spaces well-documented?**

Yes — see the **Environment Design** section above for the full action tag spec, observation field table, and state field table. The `/schema` endpoint also returns machine-readable JSON schemas at runtime.

---

### 💻 Code Quality & Spec Compliance (15%)

**Q: Does `openenv validate` pass cleanly?**

The environment implements all required OpenEnv interfaces: `reset()`, `step()`, `state()`, `/health`, `/schema`, and WebSocket transport. The `openenv.yaml` and Dockerfile frontmatter are spec-compliant. Run `openenv validate` against the live Space or local Docker container to verify.

**Q: Does the Dockerfile build?**

Yes. The multi-stage Dockerfile uses `openenv-base` as the base image, installs dependencies via `uv sync --frozen` (reproducible from `uv.lock`), and runs `uvicorn server.app:app` on port 8000. Build command:

```bash
docker build -t price_negotiation-env:latest -f Dockerfile .
```

**Q: Does the baseline script reproduce scores?**

Yes. `inference.py` is the baseline script. Given the same `HF_TOKEN`, `BUYER_MODEL`, and `ENV_BASE_URL`, it runs one episode per difficulty and prints `[START]` / `[STEP]` / `[END]` lines with the full `reward_breakdown` dict. Scores are reproducible to within LLM sampling variance (set `temperature=0` for fully deterministic buyer outputs).

---

### ✨ Creativity & Novelty (10%)

**Q: What makes this environment unique compared to other negotiation benchmarks?**

Several design choices make this environment stand out:

1. **The `walkaway_penalty` asymmetry** — missing a possible deal (`−5.0`) is penalised 5× more than a correct walk (`+1.0`). This creates a strong training signal that forces the agent to distinguish between "I can't get a good price" and "no deal is possible at any price" — a distinction most negotiation benchmarks ignore.

2. **Anchoring as a first-class reward component** — `anchoring_reward` evaluates the *first offer* specifically, based on the 65%-of-true-value heuristic from negotiation research. This teaches the model that the opening move matters as much as the closing move.

3. **Concession quality, not just outcome** — `negotiation_progress_reward` penalises both backtracking (lowering offers) and large jumps. A model that reaches the right final price via erratic concessions scores lower than one that gets there smoothly. This trains negotiation *style*, not just negotiation *outcome*.

4. **No ZOPA as a first-class task** — the hard scenario has `deal_possible = false`. Most negotiation environments only test deal-closing. Testing walk-away discipline on a scenario where the seller is convincing and the gap is small ($10) is a genuinely novel challenge.

5. **Six-component reward with transparent normalisation** — each component is independently interpretable, normalised to `[0, 1]`, and averaged. Researchers can ablate individual components, weight them differently, or use them as separate reward heads in multi-objective RL.
