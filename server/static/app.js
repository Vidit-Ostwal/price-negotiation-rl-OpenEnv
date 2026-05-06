// Same-origin API base. FastAPI serves the UI and OpenEnv endpoints together.
const API_BASE = '';

// DOM Elements
const difficultySelect = document.getElementById('difficulty');
const resetBtn = document.getElementById('reset-btn');
const stepBtn = document.getElementById('step-btn');
const mainLayout = document.getElementById('main-layout');
const chatCard = document.getElementById('chat-card');
const buyerInput = document.getElementById('buyer-input');
const productInfoBtn = document.getElementById('product-info-btn');
const productInfoModal = document.getElementById('product-info-modal');
const productInfoClose = document.getElementById('product-info-close');
const productSummary = document.getElementById('product-summary');
const productInfoContent = document.getElementById('product-info-content');
const rewardInfoBtn = document.getElementById('reward-info-btn');
const rewardInfoModal = document.getElementById('reward-info-modal');
const rewardInfoClose = document.getElementById('reward-info-close');
const rewardSummary = document.getElementById('reward-summary');
const rewardInfoContent = document.getElementById('reward-info-content');

// Quick action buttons
const actionCustom = document.getElementById('action-custom');
const actionOffer = document.getElementById('action-offer');
const actionAccept = document.getElementById('action-accept');
const actionWalk = document.getElementById('action-walk');
const offerInputRow = document.getElementById('offer-input-row');
const offerAmount = document.getElementById('offer-amount');
const offerOkBtn = document.getElementById('offer-ok-btn');

// Metrics
const metricRound = document.getElementById('metric-round');
const metricStatus = document.getElementById('metric-status');
const metricReward = document.getElementById('metric-reward');

// State
let currentState = null;
let currentRewardData = null;
let isStepPending = false;
let isEpisodeDone = true;

const REWARD_COMPONENT_LABELS = {
  surplus_reward: 'Surplus',
  walkaway_penalty: 'Walkaway Decision',
  format_reward: 'Action Format',
  efficiency_bonus: 'Efficiency',
  anchoring_reward: 'Opening Anchor',
  negotiation_progress_reward: 'Negotiation Progress',
};

const REWARD_COMPONENT_DESCRIPTIONS = {
  surplus_reward: 'Buyer surplus captured from the final accepted price.',
  walkaway_penalty: 'Whether deal or walk-away was economically correct.',
  format_reward: 'Share of buyer turns with a valid action tag.',
  efficiency_bonus: 'How quickly the negotiation closed.',
  anchoring_reward: 'Quality of the first buyer offer.',
  negotiation_progress_reward: 'Smoothness of buyer concessions over time.',
};

function setStepPending(pending) {
  isStepPending = pending;
  stepBtn.disabled = pending || isEpisodeDone;
  resetBtn.disabled = pending;
  difficultySelect.disabled = pending;
  buyerInput.disabled = pending || isEpisodeDone;
  actionCustom.disabled = pending || isEpisodeDone;
  actionOffer.disabled = pending || isEpisodeDone;
  actionAccept.disabled = pending || isEpisodeDone;
  actionWalk.disabled = pending || isEpisodeDone;
  offerAmount.disabled = pending || isEpisodeDone;
  offerOkBtn.disabled = pending || isEpisodeDone;
  stepBtn.textContent = pending ? 'Stepping...' : 'Step →';
}

function setEpisodeDone(done) {
  isEpisodeDone = done;
  stepBtn.disabled = done || isStepPending;
  buyerInput.disabled = done || isStepPending;
  actionCustom.disabled = done || isStepPending;
  actionOffer.disabled = done || isStepPending;
  actionAccept.disabled = done || isStepPending;
  actionWalk.disabled = done || isStepPending;
  offerAmount.disabled = done || isStepPending;
  offerOkBtn.disabled = done || isStepPending;
}

function formatCurrency(value, currency = 'USD') {
  if (typeof value !== 'number') return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatReward(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(3)}`;
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

function rewardTone(value) {
  if (typeof value !== 'number' || Number.isNaN(value) || value === 0) return 'neutral';
  return value > 0 ? 'positive' : 'negative';
}

function hasProductInfo(state) {
  return !!(state && state.product_info && Object.keys(state.product_info).length);
}

function updateProductInfoButton(state) {
  productInfoBtn.disabled = !hasProductInfo(state);
}

function formatLabel(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatValue(key, value, currency = 'USD') {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number' && /price|value|anchor|reserve/i.test(key)) {
    return formatCurrency(value, currency);
  }
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (Array.isArray(value)) {
    return value.every((item) => typeof item === 'number')
      ? value.map((item) => formatCurrency(item, currency)).join(' - ')
      : value.join(', ');
  }
  if (typeof value === 'object') return null;
  return String(value);
}

function appendInfoTable(parent, rows, currency = 'USD') {
  const table = document.createElement('table');
  table.className = 'info-table';

  const tbody = document.createElement('tbody');
  rows.forEach(([key, value]) => {
    const formatted = formatValue(key, value, currency);
    if (formatted === null) return;

    const row = document.createElement('tr');
    const keyCell = document.createElement('th');
    const valueCell = document.createElement('td');

    keyCell.textContent = formatLabel(key);
    valueCell.textContent = formatted;

    row.appendChild(keyCell);
    row.appendChild(valueCell);
    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  parent.appendChild(table);
}

function appendRewardMetricGrid(parent, rows) {
  const grid = document.createElement('div');
  grid.className = 'reward-metric-grid';

  rows.forEach(([label, value]) => {
    const item = document.createElement('div');
    item.className = 'reward-metric-item';

    const labelEl = document.createElement('span');
    labelEl.textContent = formatLabel(label);

    const valueEl = document.createElement('strong');
    valueEl.textContent = value ?? '—';

    item.appendChild(labelEl);
    item.appendChild(valueEl);
    grid.appendChild(item);
  });

  parent.appendChild(grid);
}

function appendTextBlock(parent, title, text) {
  if (!text) return;

  const block = document.createElement('div');
  block.className = 'info-text-block';

  const heading = document.createElement('h4');
  heading.textContent = title;

  const body = document.createElement('p');
  body.textContent = text;

  block.appendChild(heading);
  block.appendChild(body);
  parent.appendChild(block);
}

function createInfoGroup(title, subtitle, defaultOpen = false) {
  const details = document.createElement('details');
  details.className = 'info-group';
  details.open = defaultOpen;

  const summary = document.createElement('summary');
  const textWrap = document.createElement('div');
  const titleEl = document.createElement('span');
  const subtitleEl = document.createElement('small');

  titleEl.textContent = title;
  subtitleEl.textContent = subtitle;

  textWrap.appendChild(titleEl);
  textWrap.appendChild(subtitleEl);
  summary.appendChild(textWrap);
  details.appendChild(summary);

  const body = document.createElement('div');
  body.className = 'info-group-body';
  details.appendChild(body);

  return { details, body };
}

function updateRewardInfoButton() {
  rewardInfoBtn.disabled = !currentRewardData || !currentRewardData.reward_breakdown;
}

function renderProductInfo(state) {
  const info = state?.product_info || {};
  const product = info.product || {};
  const valuations = info.valuations || {};
  const metadata = info.metadata || {};
  const asymmetry = info.information_asymmetry || {};
  const currency = metadata.currency || 'USD';

  productSummary.innerHTML = '';

  const summaryItems = [
    ['Item', product.name || '—'],
    ['Category', product.category || '—'],
    ['Market price', formatCurrency(product.market_price ?? valuations.market_price, currency)],
    ['Difficulty', info.difficulty || valuations.difficulty || '—'],
    ['Buyer max', formatCurrency(valuations.buyer_true_value, currency)],
    ['Seller reserve', formatCurrency(valuations.seller_reserve_price, currency)],
    ['ZOPA', Array.isArray(valuations.zopa)
      ? valuations.zopa.map((value) => formatCurrency(value, currency)).join(' - ')
      : '—'],
    ['Deal possible', valuations.deal_possible === undefined ? '—' : String(valuations.deal_possible)],
  ];

  summaryItems.forEach(([label, value]) => {
    const item = document.createElement('div');
    item.className = 'product-summary-item';

    const labelEl = document.createElement('span');
    labelEl.textContent = label;

    const valueEl = document.createElement('strong');
    valueEl.textContent = value;

    item.appendChild(labelEl);
    item.appendChild(valueEl);
    productSummary.appendChild(item);
  });

  productInfoContent.innerHTML = '';

  if (!hasProductInfo(state)) {
    const empty = document.createElement('p');
    empty.className = 'empty-info';
    empty.textContent = 'Reset an episode to load product_info.';
    productInfoContent.appendChild(empty);
    return;
  }

  const scenarioGroup = createInfoGroup('Scenario', 'Episode and information asymmetry', true);
  appendInfoTable(scenarioGroup.body, [
    ['episode_id', info.episode_id],
    ['difficulty', info.difficulty || valuations.difficulty],
    ['buyer_context', asymmetry.buyer_context],
    ['seller_context', asymmetry.seller_context],
  ], currency);
  productInfoContent.appendChild(scenarioGroup.details);

  const productGroup = createInfoGroup('Product', 'Listing details and market context', true);
  appendInfoTable(productGroup.body, Object.entries(product), currency);
  productInfoContent.appendChild(productGroup.details);

  const valuationGroup = createInfoGroup('Valuations', 'Buyer, seller, ZOPA, and deal feasibility');
  appendInfoTable(valuationGroup.body, Object.entries(valuations), currency);
  productInfoContent.appendChild(valuationGroup.details);

  const metadataGroup = createInfoGroup('Metadata and Prompts', 'Behavior rules, limits, and raw role prompts');
  appendInfoTable(metadataGroup.body, [
    ['currency', metadata.currency],
    ['max_turns', metadata.max_turns],
    ['generator_version', metadata.generator_version],
  ], currency);

  if (metadata.buyer_behavior) {
    const heading = document.createElement('h3');
    heading.textContent = 'Buyer Behavior';
    metadataGroup.body.appendChild(heading);
    appendInfoTable(metadataGroup.body, Object.entries(metadata.buyer_behavior), currency);
  }

  if (metadata.seller_behavior) {
    const heading = document.createElement('h3');
    heading.textContent = 'Seller Behavior';
    metadataGroup.body.appendChild(heading);
    appendInfoTable(metadataGroup.body, Object.entries(metadata.seller_behavior), currency);
  }

  appendTextBlock(metadataGroup.body, 'Buyer Prompt', info.buyer_prompt);
  appendTextBlock(metadataGroup.body, 'Seller Prompt', info.seller_prompt);
  productInfoContent.appendChild(metadataGroup.details);
}

function openProductInfoModal() {
  renderProductInfo(currentState);
  productInfoModal.hidden = false;
  productInfoModal.style.display = '';
  productInfoModal.classList.add('open');
  productInfoModal.setAttribute('aria-hidden', 'false');
  productInfoClose.focus();
}

function closeProductInfoModal() {
  productInfoModal.classList.remove('open');
  productInfoModal.setAttribute('aria-hidden', 'true');
  productInfoModal.hidden = true;
  productInfoModal.style.display = 'none';
  productInfoBtn.focus();
}

function renderRewardInfo() {
  const data = currentRewardData || {};
  const reward = data.reward;
  const components = data.reward_components || {};
  const breakdown = data.reward_breakdown || {};
  const weights = data.reward_weights || {};
  const tone = rewardTone(reward);

  rewardSummary.innerHTML = '';
  const summaryItems = [
    ['Score', formatReward(reward), tone],
    ['Sign', tone === 'positive' ? 'Positive' : tone === 'negative' ? 'Negative' : 'Neutral', tone],
    ['Components', String(Object.keys(breakdown).length || '—'), 'neutral'],
    ['Total Weight', formatPercent(Object.values(weights).reduce((sum, value) => sum + value, 0)), 'neutral'],
  ];

  summaryItems.forEach(([label, value, itemTone]) => {
    const item = document.createElement('div');
    item.className = `product-summary-item reward-${itemTone}`;

    const labelEl = document.createElement('span');
    labelEl.textContent = label;

    const valueEl = document.createElement('strong');
    valueEl.textContent = value;

    item.appendChild(labelEl);
    item.appendChild(valueEl);
    rewardSummary.appendChild(item);
  });

  rewardInfoContent.innerHTML = '';
  if (!Object.keys(breakdown).length) {
    const empty = document.createElement('p');
    empty.className = 'empty-info';
    empty.textContent = 'Take a step to compute reward breakdown details.';
    rewardInfoContent.appendChild(empty);
    return;
  }

  Object.entries(breakdown).forEach(([name, raw]) => {
    const component = components[name] || {};
    const card = document.createElement('div');
    card.className = `reward-component reward-${rewardTone(raw)}`;

    const header = document.createElement('div');
    header.className = 'reward-component-header';

    const titleWrap = document.createElement('div');
    const title = document.createElement('h3');
    title.textContent = REWARD_COMPONENT_LABELS[name] || formatLabel(name);
    const description = document.createElement('p');
    description.textContent = REWARD_COMPONENT_DESCRIPTIONS[name] || 'Reward component score.';
    titleWrap.appendChild(title);
    titleWrap.appendChild(description);

    const rawScore = document.createElement('strong');
    rawScore.textContent = formatReward(raw);

    header.appendChild(titleWrap);
    header.appendChild(rawScore);
    card.appendChild(header);

    appendRewardMetricGrid(card, [
      ['raw_score', formatReward(raw)],
      ['component_score', formatReward(component.score)],
      ['weight', formatPercent(component.weight ?? weights[name])],
      ['weighted_contribution', formatReward(component.weighted_score)],
    ]);

    rewardInfoContent.appendChild(card);
  });
}

function openRewardInfoModal() {
  renderRewardInfo();
  rewardInfoModal.hidden = false;
  rewardInfoModal.style.display = '';
  rewardInfoModal.classList.add('open');
  rewardInfoModal.setAttribute('aria-hidden', 'false');
  rewardInfoClose.focus();
}

function closeRewardInfoModal() {
  rewardInfoModal.classList.remove('open');
  rewardInfoModal.setAttribute('aria-hidden', 'true');
  rewardInfoModal.hidden = true;
  rewardInfoModal.style.display = 'none';
  rewardInfoBtn.focus();
}

// Update metrics from observation
function updateMetricsFromResponse(data) {
  const obs = data.observation || {};
  const done = data.done ?? obs.done ?? false;
  currentRewardData = {
    reward: data.reward ?? obs.reward,
    reward_breakdown: data.reward_breakdown ?? obs.reward_breakdown,
    reward_weights: data.reward_weights ?? obs.reward_weights,
    reward_components: data.reward_components ?? obs.reward_components,
  };
  metricRound.textContent = obs.negotiation_round ?? '—';
  metricStatus.textContent = obs.deal_status ?? '—';
  metricReward.textContent = formatReward(currentRewardData.reward);
  metricReward.className = `metric-value reward-value reward-${rewardTone(currentRewardData.reward)}`;
  updateRewardInfoButton();
  setEpisodeDone(done);
  
  // Update status card color
  const statusCard = metricStatus.closest('.metric-card');
  statusCard.classList.remove('status-ongoing', 'status-accepted', 'status-walked');
  
  if (obs.deal_status) {
    const status = obs.deal_status.toLowerCase();
    if (status.includes('accept')) {
      statusCard.classList.add('status-accepted');
    } else if (status.includes('walk')) {
      statusCard.classList.add('status-walked');
    } else {
      statusCard.classList.add('status-ongoing');
    }
  }
}

// Render chat messages
function renderChat(state) {
  chatCard.innerHTML = '';

  const product = state?.product_info?.product || {};
  if (product.description) {
    const productCard = document.createElement('div');
    productCard.className = 'chat-product-card';

    const label = document.createElement('div');
    label.className = 'chat-product-label';
    label.textContent = 'Product Context';

    const title = document.createElement('div');
    title.className = 'chat-product-title';
    title.textContent = product.name || 'Current product';

    const description = document.createElement('p');
    description.textContent = product.description;

    productCard.appendChild(label);
    productCard.appendChild(title);
    productCard.appendChild(description);
    chatCard.appendChild(productCard);
  }

  if (!state || !state.buyer_messages || state.buyer_messages.length === 0) {
    return;
  }
  
  state.buyer_messages.forEach((msg) => {
    const role = (msg.role || '').toLowerCase();
    if (role === 'system') return;

    const participant = role === 'assistant' || role === 'buyer' ? 'buyer' : 'seller';
    
    const row = document.createElement('div');
    row.className = `message-row ${participant}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = participant === 'buyer' ? 'BY' : 'SE';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = msg.content;
    
    row.appendChild(avatar);
    row.appendChild(bubble);
    chatCard.appendChild(row);
  });
  
  // Scroll to bottom
  chatCard.scrollTop = chatCard.scrollHeight;
}

// Reset episode
async function resetEpisode() {
  setEpisodeDone(true);
  stepBtn.disabled = true;
  resetBtn.disabled = true;
  
  try {
    const difficulty = difficultySelect.value;
    const params = difficulty !== 'any' ? `?difficulty=${difficulty}` : '';
    
    const response = await fetch(`${API_BASE}/reset${params}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) throw new Error(`Reset failed: ${response.statusText}`);
    
    const data = await response.json();
    
    updateMetricsFromResponse(data);
    // Sync state after reset so chat and product info update automatically.
    await refreshState();
    
    mainLayout.hidden = false;
    mainLayout.style.display = '';
    mainLayout.classList.remove('hidden');
    buyerInput.value = '';
  } catch (error) {
    mainLayout.classList.add('hidden');
    mainLayout.hidden = true;
    mainLayout.style.display = 'none';
    setEpisodeDone(true);
    console.error('Reset error:', error);
  } finally {
    stepBtn.disabled = isEpisodeDone || isStepPending;
    resetBtn.disabled = false;
  }
}

// Step episode
async function stepEpisode(responseText) {
  if (isStepPending || isEpisodeDone) return;

  const buyerResponse = (responseText ?? buyerInput.value).trim();
  if (!buyerResponse) {
    return;
  }
  
  setStepPending(true);
  
  try {
    const response = await fetch(`${API_BASE}/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action: { buyer_response: buyerResponse }
      })
    });
    
    if (!response.ok) throw new Error(`Step failed: ${response.statusText}`);
    
    const data = await response.json();
    
    updateMetricsFromResponse(data);
    // Sync state after step so chat and product info update automatically.
    await refreshState();
    
    buyerInput.value = '';
    offerInputRow.classList.remove('open');
    offerAmount.value = '';
  } catch (error) {
    console.error('Step error:', error);
  } finally {
    setStepPending(false);
  }
}

// Refresh state
async function refreshState() {
  try {
    const response = await fetch(`${API_BASE}/state`);
    if (!response.ok) throw new Error(`State fetch failed: ${response.statusText}`);
    
    currentState = await response.json();
    updateProductInfoButton(currentState);
    renderChat(currentState);
    
    return currentState;
  } catch (error) {
    console.error('State error:', error);
    return null;
  }
}

// Quick action: Custom
actionCustom.addEventListener('click', () => {
  if (isStepPending || isEpisodeDone) return;
  offerInputRow.classList.remove('open');
  buyerInput.focus();
});

// Quick action: Offer
actionOffer.addEventListener('click', () => {
  if (isStepPending || isEpisodeDone) return;
  offerInputRow.classList.toggle('open');
  if (offerInputRow.classList.contains('open')) {
    offerAmount.focus();
  }
});

function insertOffer() {
  const amount = offerAmount.value.trim();
  if (!amount || isStepPending || isEpisodeDone) return;
  
  stepEpisode(`I'd like to offer $${amount}. <action>OFFER $${amount}</action>`);
}

offerOkBtn.addEventListener('click', insertOffer);
offerAmount.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') insertOffer();
});

// Quick action: Accept
actionAccept.addEventListener('click', () => {
  stepEpisode('I accept the deal. <action>ACCEPT</action>');
});

// Quick action: Walk
actionWalk.addEventListener('click', () => {
  stepEpisode("I'm walking away. <action>WALK</action>");
});

productInfoBtn.addEventListener('click', openProductInfoModal);
productInfoClose.addEventListener('click', closeProductInfoModal);
productInfoModal.addEventListener('click', (e) => {
  if (e.target === productInfoModal) closeProductInfoModal();
});

rewardInfoBtn.addEventListener('click', openRewardInfoModal);
rewardInfoClose.addEventListener('click', closeRewardInfoModal);
rewardInfoModal.addEventListener('click', (e) => {
  if (e.target === rewardInfoModal) closeRewardInfoModal();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && productInfoModal.classList.contains('open')) {
    closeProductInfoModal();
  }
  if (e.key === 'Escape' && rewardInfoModal.classList.contains('open')) {
    closeRewardInfoModal();
  }
});

// Event listeners
resetBtn.addEventListener('click', resetEpisode);
stepBtn.addEventListener('click', stepEpisode);

buyerInput.addEventListener('keydown', (e) => {
  // Ctrl/Cmd + Enter to submit
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    stepEpisode();
  }
});

setEpisodeDone(true);

