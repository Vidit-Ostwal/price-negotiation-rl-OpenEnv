# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradio UI for price negotiation — redesigned with CSS/JS injection.

Drop-in replacement for the original gradio_ui.py.
``server/app.py`` passes ``gradio_builder=build_custom_gradio_ui`` unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import gradio as gr

from openenv.core.env_server.serialization import serialize_observation


# ---------------------------------------------------------------------------
# CSS — injected via gr.HTML
# ---------------------------------------------------------------------------

INJECTED_CSS = '''
<style id="neg-styles">
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
  --primary:#2563EB;
  --primary-d:#1D4ED8;
  --primary-l:#DBEAFE;
  --success:#16A34A;
  --success-l:#DCFCE7;
  --danger:#DC2626;
  --danger-l:#FEE2E2;
  --warning:#D97706;
  --warning-l:#FEF3C7;
  --surface:#F8FAFC;
  --card:#FFFFFF;
  --bd:#E2E8F0;
  --bd-mid:#CBD5E1;
  --t1:#0F172A;
  --t2:#334155;
  --t3:#64748B;
  --r:12px;
  --rl:16px;
}
body,.gradio-container{font-family:'DM Sans',sans-serif!important;background:var(--surface)!important;color:var(--t1)!important;}
footer{display:none!important;}
#neg-header-html{background:linear-gradient(135deg,var(--primary),var(--primary-d));border-radius:var(--rl);padding:1.1rem 1.4rem;margin-bottom:.5rem;color:#fff;}
.nm-card,#neg-actions-html,#chat-md,#obs-md{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);box-shadow:0 6px 18px rgba(15,23,42,.06);}
#step-btn button,#reset-btn button{background:var(--primary)!important;color:#fff!important;border:none!important;border-radius:var(--r)!important;}
#step-btn button:hover,#reset-btn button:hover{background:var(--primary-d)!important;}
#refresh-btn button{border:1px solid var(--bd-mid)!important;border-radius:var(--r)!important;}
.br.buyer .bb{background:var(--primary);color:#fff;}
.na-btn.offer .na-icon{background:var(--primary-l);color:var(--primary);}
.na-btn.accept .na-icon{background:var(--success-l);color:var(--success);}
.na-btn.walk .na-icon{background:var(--danger-l);color:var(--danger);}
</style>
'''

# ---------------------------------------------------------------------------
# Static HTML blocks rendered as gr.HTML() — stable DOM anchors
# ---------------------------------------------------------------------------

HEADER_HTML = """
<div id="neg-header-html">
  <div class="nh-logo">&#x1F91D;</div>
  <div>
    <h1>Price Negotiation</h1>
    <p>Play as the buyer — reset an episode, make offers, and close the deal.</p>
  </div>
</div>
"""

METRICS_HTML = """
<div id="neg-metrics-html">
  <div class="nm-card" id="nm-round">
    <div class="nm-label">Round</div>
    <div class="nm-value">&#8212;</div>
  </div>
  <div class="nm-card" id="nm-status">
    <div class="nm-label">Deal status</div>
    <div class="nm-value" style="font-size:13px;line-height:1.4;">&#8212;</div>
  </div>
  <div class="nm-card" id="nm-reward">
    <div class="nm-label">Reward</div>
    <div class="nm-value">&#8212;</div>
  </div>
  <div class="nm-card" id="nm-done">
    <div class="nm-label">Done</div>
    <div class="nm-value">&#8212;</div>
  </div>
</div>
"""

ACTIONS_HTML = """
<div id="neg-actions-html">
  <div class="na-title">Quick actions</div>
  <button class="na-btn offer" id="na-btn-offer">
    <span class="na-icon">$</span>
    <span>
      <span style="display:block;">Make an offer</span>
      <span class="na-sub">Set your price</span>
    </span>
  </button>
  <div id="na-offer-row">
    <span style="font-size:13px;color:var(--teal-d);font-weight:600;flex-shrink:0;">$</span>
    <input id="na-offer-amt" type="number" placeholder="Enter amount" min="0" />
    <button id="na-offer-ok">Insert</button>
  </div>
  <button class="na-btn accept" id="na-btn-accept">
    <span class="na-icon">&#10003;</span>
    <span>
      <span style="display:block;">Accept deal</span>
      <span class="na-sub">Agree to current price</span>
    </span>
  </button>
  <button class="na-btn walk" id="na-btn-walk">
    <span class="na-icon">&#10005;</span>
    <span>
      <span style="display:block;">Walk away</span>
      <span class="na-sub">End negotiation</span>
    </span>
  </button>
</div>
"""

DEBUG_TOGGLE_HTML = """
<div id="neg-debug-toggle">
  <span id="neg-debug-arrow" style="font-size:10px;display:inline-block;transition:transform 0.2s;">&#9658;</span>
  <span>Debug &mdash; raw JSON</span>
</div>
"""

# ---------------------------------------------------------------------------
# JS — polling-based (avoids MutationObserver infinite loop on DOM writes)
# ---------------------------------------------------------------------------

INJECTED_JS = r"""
<script id="neg-js">
(function () {
  'use strict';

  // ── Fill Gradio textarea (bypasses React/Svelte synthetic event gate) ─────
  function fillTextarea(elemId, value) {
    const wrap = document.getElementById(elemId);
    if (!wrap) return;
    const ta = wrap.querySelector('textarea');
    if (!ta) return;
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
    setter.call(ta, value);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
  }

  // ── Metric card helper ────────────────────────────────────────────────────
  function setMetric(id, value, cls) {
    const card = document.getElementById(id);
    if (!card) return;
    const el = card.querySelector('.nm-value');
    if (el) el.textContent = value;
    card.classList.remove('is-teal', 'is-coral', 'is-amber');
    if (cls) card.classList.add(cls);
  }

  function parseMetrics(text) {
    const g = (re) => { const m = text.match(re); return m ? m[1].trim() : null; };
    const round  = g(/Round[^:]*:\s*`?(\w+)`?/);
    const status = g(/Deal status[^:]*:\s*`?([^`\n\r]+)`?/);
    const reward = g(/Reward[^:]*:\s*`?([^`\n\r]+)`?/);
    const done   = g(/Done[^:]*:\s*`?([^`\n\r]+)`?/);
    if (round)  setMetric('nm-round',  round, null);
    if (reward) setMetric('nm-reward', reward, null);
    if (done)   setMetric('nm-done',   done, done === 'True' ? 'is-teal' : null);
    if (status) {
      const s = status.toLowerCase();
      const cls = s.includes('accept') ? 'is-teal'
                : (s.includes('walk') || s.includes('rej')) ? 'is-coral'
                : 'is-amber';
      setMetric('nm-status', status, cls);
    }
  }

  // ── Chat bubble renderer ──────────────────────────────────────────────────
  const SELLER = new Set(['user', 'seller']);
  const BUYER  = new Set(['assistant', 'buyer']);

  function renderBubbles(container) {
    // container is the Gradio markdown wrapper (div with class prose etc.)
    const raw = container.innerText || '';
    if (!raw.trim()) return;

    const lines = raw.split('\n');
    const bubbles = [];
    for (const line of lines) {
      const t = line.trim();
      const ci = t.indexOf(':');
      if (ci < 1 || ci > 25) continue;
      const role = t.slice(0, ci).trim().toLowerCase();
      const content = t.slice(ci + 1).trim();
      if (!content) continue;
      // Skip markdown headings / metadata lines
      if (t.startsWith('#') || t.startsWith('_') || t.startsWith('*')) continue;
      // Skip lines where "role" is a long phrase (section labels)
      if (role.split(' ').length > 2) continue;
      let type = 'sys';
      if (BUYER.has(role))  type = 'buyer';
      if (SELLER.has(role)) type = 'seller';
      bubbles.push({ display: t.slice(0, ci).trim(), type, content });
    }
    if (!bubbles.length) return;

    const wrap = document.createElement('div');
    wrap.className = 'bw';
    for (const b of bubbles) {
      const row = document.createElement('div');
      row.className = 'br ' + b.type;
      if (b.type !== 'sys') {
        const av = document.createElement('div');
        av.className = 'bav';
        av.textContent = b.display.slice(0, 2).toUpperCase();
        row.appendChild(av);
      }
      const bb = document.createElement('div');
      bb.className = 'bb';
      bb.textContent = b.content;
      row.appendChild(bb);
      wrap.appendChild(row);
    }

    container.innerHTML = '';
    container.appendChild(wrap);
    container.scrollTop = container.scrollHeight;
  }

  // ── Wire action buttons ───────────────────────────────────────────────────
  function wireActions() {
    const offerBtn  = document.getElementById('na-btn-offer');
    if (!offerBtn || offerBtn._wired) return;
    offerBtn._wired = true;

    const offerRow = document.getElementById('na-offer-row');
    const offerAmt = document.getElementById('na-offer-amt');
    const offerOk  = document.getElementById('na-offer-ok');

    offerBtn.addEventListener('click', () => {
      offerRow.classList.toggle('open');
      if (offerRow.classList.contains('open')) offerAmt.focus();
    });

    function insertOffer() {
      const v = offerAmt.value;
      if (!v) return;
      fillTextarea('buyer-input', "I'd like to offer $" + v + ". <action>OFFER $" + v + "</action>");
      offerRow.classList.remove('open');
      offerAmt.value = '';
    }
    offerOk.addEventListener('click', insertOffer);
    offerAmt.addEventListener('keydown', (e) => { if (e.key === 'Enter') insertOffer(); });

    document.getElementById('na-btn-accept').addEventListener('click', () => {
      fillTextarea('buyer-input', "I accept the deal. <action>ACCEPT</action>");
    });
    document.getElementById('na-btn-walk').addEventListener('click', () => {
      fillTextarea('buyer-input', "I'm walking away. <action>WALK</action>");
    });
  }

  // ── Wire debug toggle ─────────────────────────────────────────────────────
  function wireDebugToggle() {
    const toggle = document.getElementById('neg-debug-toggle');
    if (!toggle || toggle._wired) return;
    toggle._wired = true;
    toggle.addEventListener('click', () => {
      const raw   = document.getElementById('raw-json');
      const arrow = document.getElementById('neg-debug-arrow');
      if (!raw) return;
      const open = raw.classList.toggle('neg-open');
      if (arrow) arrow.style.transform = open ? 'rotate(90deg)' : '';
    });
  }

  // ── Main poll loop ────────────────────────────────────────────────────────
  // Gradio re-renders on every update; we poll instead of MutationObserver
  // to avoid infinite loops when we write to the DOM ourselves.

  let lastObsText  = '';
  let lastChatText = '';

  function poll() {
    wireActions();
    wireDebugToggle();

    // Obs metrics
    const obsEl = document.getElementById('obs-md');
    if (obsEl) {
      const inner = obsEl.querySelector('.prose') || obsEl.querySelector('[data-testid="markdown"]') || obsEl;
      const text = inner.innerText || '';
      if (text && text !== lastObsText) {
        lastObsText = text;
        parseMetrics(text);
      }
    }

    // Chat bubbles
    const chatEl = document.getElementById('chat-md');
    if (chatEl) {
      const inner = chatEl.querySelector('.prose') || chatEl.querySelector('[data-testid="markdown"]') || chatEl;
      // Don't rewrite if already has bubbles
      if (inner.querySelector('.bw')) return;
      const text = inner.innerText || '';
      if (text && text !== lastChatText) {
        lastChatText = text;
        renderBubbles(inner);
      }
    }
  }

  setInterval(poll, 400);
  document.addEventListener('DOMContentLoaded', poll);
  setTimeout(poll, 800);
})();
</script>
"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_observation_summary(payload: Dict[str, Any]) -> str:
    obs = payload.get("observation") or {}
    parts: List[str] = []
    if obs.get("deal_status"):
        parts.append(f"**Deal status:** `{obs['deal_status']}`")
    if obs.get("next_turn") is not None:
        parts.append(f"**Next turn:** `{obs['next_turn']}`")
    if obs.get("negotiation_round") is not None:
        parts.append(f"**Round:** `{obs['negotiation_round']}`")
    done = payload.get("done")
    if done is not None:
        parts.append(f"**Done:** `{done}`")
    reward = payload.get("reward")
    if reward is not None:
        parts.append(f"**Reward:** `{reward}`")
    return "\n\n".join(parts) if parts else "*No observation fields in response.*"


def _format_product_line(state: Dict[str, Any]) -> str:
    info = state.get("product_info") or {}
    product = info.get("product") or {}
    name = product.get("name", "Unknown item")
    diff = info.get("difficulty") or (info.get("valuations") or {}).get("difficulty", "")
    suffix = f" — _{diff}_" if diff else ""
    return f"**Scenario:** {name}{suffix}"


def _format_message_list(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
        lines.append("")
    return "\n".join(lines) if lines else "_No messages yet._"


def _format_state_markdown(state: Dict[str, Any]) -> str:
    # Use buyer_messages — it contains the full dialogue from buyer POV
    # Roles will be "assistant" (buyer) and "user" (seller)
    msgs = _format_message_list(state.get("buyer_messages") or [])
    return msgs


# ---------------------------------------------------------------------------
# Reset helper (unchanged)
# ---------------------------------------------------------------------------

async def _reset_with_difficulty(web_manager: Any, difficulty: str) -> Dict[str, Any]:
    if difficulty in ("easy", "medium", "hard"):
        observation = await web_manager._run_sync_in_thread_pool(
            web_manager.env.reset,
            difficulty=difficulty,
        )
    else:
        observation = await web_manager._run_sync_in_thread_pool(web_manager.env.reset)

    state = web_manager.env.state
    serialized = serialize_observation(observation)
    web_manager.episode_state.episode_id = state.episode_id
    web_manager.episode_state.step_count = 0
    web_manager.episode_state.current_observation = serialized["observation"]
    web_manager.episode_state.action_logs = []
    web_manager.episode_state.is_reset = True
    await web_manager._send_state_update()
    return serialized


# ---------------------------------------------------------------------------
# Main Gradio builder
# ---------------------------------------------------------------------------

def build_custom_gradio_ui(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[Any],
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    """Build Gradio Blocks for the negotiation UI (OpenEnv ``gradio_builder`` hook)."""
    del action_fields, is_chat_env

    display_name = getattr(metadata, "name", None) or title or "Price negotiation"

    async def on_reset(difficulty: str):
        try:
            data = await _reset_with_difficulty(web_manager, difficulty)
            st = web_manager.get_state()
            return (
                _format_observation_summary(data),
                _format_state_markdown(st),
                json.dumps(data, indent=2),
                f"Reset OK — {difficulty}",
            )
        except Exception as e:
            return "", "", "", f"Reset error: {e}"

    async def on_step(buyer_text: str):
        if not str(buyer_text).strip():
            return "", "", "", "Enter a buyer message before stepping."
        try:
            data = await web_manager.step_environment(
                {"buyer_response": str(buyer_text).strip()}
            )
            st = web_manager.get_state()
            return (
                _format_observation_summary(data),
                _format_state_markdown(st),
                json.dumps(data, indent=2),
                "Step OK.",
            )
        except Exception as e:
            return "", "", "", f"Step error: {e}"

    def on_refresh_state():
        try:
            st = web_manager.get_state()
            return (
                _format_state_markdown(st),
                json.dumps(st, indent=2),
                "State refreshed.",
            )
        except Exception as e:
            return "", "", f"State error: {e}"

    with gr.Blocks(title=display_name) as demo:

        # ① CSS — first thing, so styles exist before elements render
        gr.HTML(INJECTED_CSS)

        # ② Header — stable gr.HTML anchor (no Gradio elem_id needed)
        gr.HTML(HEADER_HTML)

        # ③ Metric cards — stable gr.HTML anchor, updated by JS poll
        gr.HTML(METRICS_HTML)

        # ④ Quick-start
        if quick_start_md:
            with gr.Accordion("Quick start", open=False):
                gr.Markdown(quick_start_md)

        # ⑤ Controls row
        with gr.Row():
            difficulty = gr.Dropdown(
                elem_id="difficulty-dropdown",
                label="Difficulty",
                choices=[
                    ("Cycle all scenarios", "any"),
                    ("Easy", "easy"),
                    ("Medium", "medium"),
                    ("Hard", "hard"),
                ],
                value="any",
                scale=2,
            )
            reset_btn = gr.Button(
                "Reset episode",
                elem_id="reset-btn",
                variant="secondary",
                scale=1,
            )
            refresh_btn = gr.Button(
                "Refresh state",
                elem_id="refresh-btn",
                variant="secondary",
                scale=1,
            )

        # ⑥ Status
        status = gr.Textbox(
            elem_id="status-box",
            label="Status",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        # ⑦ Two-column layout
        with gr.Row(equal_height=False):

            # Left: action panel (gr.HTML — no Gradio component instability)
            with gr.Column(scale=1, min_width=210):
                gr.HTML(ACTIONS_HTML)

            # Right: obs + chat + input
            with gr.Column(scale=3):
                obs_md = gr.Markdown(
                    "*Reset to begin.*",
                    elem_id="obs-md",
                )
                chat_md = gr.Markdown(
                    "",
                    elem_id="chat-md",
                )
                buyer_input = gr.Textbox(
                    elem_id="buyer-input",
                    label="Buyer response",
                    lines=3,
                    placeholder=(
                        "Type your reply, or use the quick-action buttons.\n"
                        "Tags: <action>OFFER $X</action>  "
                        "<action>ACCEPT</action>  <action>WALK</action>"
                    ),
                )
                step_btn = gr.Button(
                    "Step \u2192",
                    elem_id="step-btn",
                    variant="primary",
                )

        # ⑧ Debug toggle + raw JSON
        gr.HTML(DEBUG_TOGGLE_HTML)
        raw_json = gr.Code(
            elem_id="raw-json",
            label="Last reset/step JSON",
            language="json",
            interactive=False,
        )

        # ⑨ JS — last, so all DOM elements exist when script runs
        gr.HTML(INJECTED_JS)

        # ── Event wiring (unchanged contract) ──
        reset_btn.click(
            fn=on_reset,
            inputs=[difficulty],
            outputs=[obs_md, chat_md, raw_json, status],
        )
        step_btn.click(
            fn=on_step,
            inputs=[buyer_input],
            outputs=[obs_md, chat_md, raw_json, status],
        )
        refresh_btn.click(
            fn=on_refresh_state,
            outputs=[chat_md, raw_json, status],
        )

    return demo