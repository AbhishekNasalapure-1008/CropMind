/* ═══════════════════════════════════════════════════════════════
   CropMind — results.js
   Handles: dynamic results rendering, SVG ring animations,
            heatmap toggle, accordion, confidence bar
   ═══════════════════════════════════════════════════════════════ */

'use strict';

const CIRCUMFERENCE = 220; // stroke-dasharray for r=35 circle

/* ── Helpers ─────────────────────────────────────────────────── */
function capitalize(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : '';
}

function severityClass(sev) {
  const map = { none: 'severity-none', low: 'severity-low', moderate: 'severity-moderate', severe: 'severity-severe' };
  return map[sev] || 'severity-none';
}

function urgencyClass(urgency) {
  const u = (urgency || '').toLowerCase();
  if (u.includes('immediate')) return 'urgency-immediate';
  if (u.includes('week')) return 'urgency-week';
  return 'urgency-monitor';
}

function nutrientColor(nutrient) {
  const map = { nitrogen: '#FCD34D', phosphorus: '#C084FC', potassium: '#FB923C' };
  return map[nutrient] || '#4ADE80';
}

/* ── Ring animation ───────────────────────────────────────────── */
function animateRing(ringEl, pct, delay = 0) {
  const offset = CIRCUMFERENCE - (pct / 100) * CIRCUMFERENCE;
  setTimeout(() => {
    ringEl.style.strokeDashoffset = offset;
  }, delay);
}

/* ── Build NPK Deficiency Card ────────────────────────────────── */
function buildDeficiencyCard(data) {
  const { deficiencies, primary_deficiency, severity, confidence } = data;
  const N = Math.round((deficiencies.nitrogen || 0) * 100);
  const P = Math.round((deficiencies.phosphorus || 0) * 100);
  const K = Math.round((deficiencies.potassium || 0) * 100);
  const conf = Math.round((confidence || 0) * 100);

  return `
    <div class="result-card" id="deficiency-card">
      <div class="card-title"><span>📊</span> NPK Deficiency Analysis</div>
      ${data.demo_mode ? '<div class="demo-banner">⚡ Demo Mode — showing simulated results (train model for real predictions)</div>' : ''}

      <div class="npk-rings">
        ${[
      { label: 'N', name: 'Nitrogen', cls: 'n', pct: N },
      { label: 'P', name: 'Phosphorus', cls: 'p', pct: P },
      { label: 'K', name: 'Potassium', cls: 'k', pct: K },
    ].map(({ label, name, cls, pct }) => `
          <div class="npk-ring-wrap">
            <div class="npk-svg-container">
              <svg class="npk-ring-svg" viewBox="0 0 90 90" aria-label="${name} deficiency: ${pct}%">
                <circle class="ring-bg"   cx="45" cy="45" r="35"/>
                <circle class="ring-fill ${cls}" cx="45" cy="45" r="35"
                  style="stroke-dasharray:${CIRCUMFERENCE};stroke-dashoffset:${CIRCUMFERENCE};"
                  data-pct="${pct}"/>
              </svg>
              <div class="ring-label-wrap" role="img" aria-label="${pct}%">
                <span class="ring-pct" id="ring-pct-${cls}">0%</span>
              </div>
            </div>
            <span class="ring-nutrient ${cls}">${label}</span>
            <small>${name}</small>
          </div>
        `).join('')}
      </div>

      <div class="primary-diagnosis">
        <div>
          <div style="font-size:0.72rem;color:var(--text-muted);margin-bottom:4px;font-family:var(--font-mono);text-transform:uppercase;letter-spacing:0.08em;">Primary Deficiency</div>
          <div class="diagnosis-name" style="color:${nutrientColor(primary_deficiency)}">${capitalize(primary_deficiency)} Deficiency</div>
        </div>
        <span class="severity-badge ${severityClass(severity)}" role="status" aria-label="Severity: ${severity}">${capitalize(severity)}</span>
      </div>

      <div class="confidence-wrap">
        <div class="confidence-label">
          <span>AI Confidence</span>
          <strong id="conf-value">0%</strong>
        </div>
        <div class="confidence-bar" role="progressbar" aria-valuenow="${conf}" aria-valuemin="0" aria-valuemax="100">
          <div class="confidence-fill" id="confidence-fill"></div>
        </div>
      </div>
    </div>
  `;
}

/* ── Build Heatmap Card ───────────────────────────────────────── */
function buildHeatmapCard(data, originalSrc) {
  const heatmapSrc = data.heatmap_base64
    ? `data:image/png;base64,${data.heatmap_base64}`
    : '';

  // Store srcs on the container element via data attrs so the
  // event listener (wired in renderResults) can read them
  return `
    <div class="result-card heatmap-card"
         data-heatmap-src="${heatmapSrc}"
         data-original-src="${originalSrc}">
      <div class="card-title"><span>🔬</span> Visual Explanation (Grad-CAM)</div>
      <div class="heatmap-toggles" role="group" aria-label="Image display toggle">
        <button class="heatmap-btn active" data-mode="heatmap" aria-pressed="true">Heatmap Overlay</button>
        <button class="heatmap-btn" data-mode="original" aria-pressed="false">Original</button>
        <button class="heatmap-btn" data-mode="side" aria-pressed="false">Side by Side</button>
      </div>

      <div class="heatmap-display" id="heatmap-display">
        <img src="${heatmapSrc || '/static/assets/leaf-icon.svg'}"
             alt="Heatmap showing AI attention areas on the leaf"
             id="heatmap-img"
             style="width:100%;border-radius:12px;display:block;"/>
      </div>
      <div class="color-scale" aria-label="Color scale legend">
        <span>Low</span>
        <div class="color-scale-bar" role="img" aria-label="Color gradient from blue (low attention) to red (high attention)"></div>
        <span>High Attention</span>
      </div>
    </div>
  `;
}

/* ── Wire up heatmap toggle buttons (called from renderResults) ── */
function wireHeatmapToggles(panel) {
  const card = panel.querySelector('.heatmap-card');
  if (!card) return;
  const heatmapSrc = card.dataset.heatmapSrc || '';
  const originalSrc = card.dataset.originalSrc || '';
  const modes = { heatmap: heatmapSrc, original: originalSrc };

  card.querySelectorAll('.heatmap-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      card.querySelectorAll('.heatmap-btn').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      });
      this.classList.add('active');
      this.setAttribute('aria-pressed', 'true');

      const mode = this.dataset.mode;
      const display = document.getElementById('heatmap-display');
      if (!display) return;

      if (mode === 'side') {
        display.style.display = 'grid';
        display.style.gridTemplateColumns = '1fr 1fr';
        display.style.gap = '8px';
        display.innerHTML = `
                  <div>
                    <img src="${originalSrc}" alt="Original leaf" style="width:100%;border-radius:8px;"/>
                    <p style="font-size:0.7rem;text-align:center;margin-top:4px;color:var(--text-muted)">Original</p>
                  </div>
                  <div>
                    <img src="${heatmapSrc}" alt="Heatmap overlay" style="width:100%;border-radius:8px;"/>
                    <p style="font-size:0.7rem;text-align:center;margin-top:4px;color:var(--text-muted)">Heatmap</p>
                  </div>`;
      } else {
        display.style.display = '';
        display.style.gridTemplateColumns = '';
        const src = modes[mode] || modes.heatmap;
        display.innerHTML = `<img src="${src}" alt="${mode} view" id="heatmap-img" style="width:100%;border-radius:12px;display:block;"/>`;
      }
    });
  });
}

/* ── Build Fertilizer Card ────────────────────────────────────── */
function buildFertilizerCard(data) {
  const recs = data.recommendations || [];

  const recHTML = recs.map(rec => `
    <div class="rec-item">
      <div class="rec-header">
        <div class="rec-name">🌿 ${rec.fertilizer}</div>
        <span class="urgency-badge ${urgencyClass(rec.urgency)}" role="status">${rec.urgency}</span>
      </div>
      <div class="rec-details">
        <div class="rec-detail"><strong>Rate:</strong> ${rec.rate}</div>
        <div class="rec-detail"><strong>Method:</strong> ${rec.method}</div>
      </div>
      <button class="accordion-trigger" aria-expanded="false" aria-controls="acc-body-${rec.nutrient}">
        Why this recommendation? <span class="accordion-arrow" aria-hidden="true">▾</span>
      </button>
      <div class="accordion-body" id="acc-body-${rec.nutrient}" role="region">
        ${rec.reason}
      </div>
    </div>
  `).join('');

  return `
    <div class="result-card">
      <div class="card-title"><span>💊</span> Fertilizer Recommendations</div>
      ${recHTML || '<p style="font-size:0.85rem;color:var(--text-muted)">No critical deficiencies detected — continue regular nutrient management.</p>'}
    </div>
  `;
}

/* ── Build Agronomic Insight Card ─────────────────────────────── */
function buildInsightCard(data) {
  const symptoms = data.symptoms_matched || [];
  const tips = data.preventive_tips || [];
  const insight = data.agronomic_insight || '';

  const symptomsHTML = symptoms.map(s => `<li>${s}</li>`).join('');
  const tipsHTML = tips.map((t, i) =>
    `<li class="tip-item"><span class="tip-num" aria-hidden="true">0${i + 1}</span>${t}</li>`
  ).join('');

  return `
    <div class="result-card">
      <div class="card-title"><span>🌱</span> Agronomic Insight</div>
      ${insight ? `<p class="insight-text">${insight}</p>` : ''}

      ${symptoms.length ? `
        <div class="nutrient-section-title" style="margin-bottom:8px">Symptoms Matched</div>
        <ul class="symptoms-list" aria-label="Detected symptoms">${symptomsHTML}</ul>
      ` : ''}

      ${tips.length ? `
        <div class="nutrient-section-title" style="margin-bottom:8px">Preventive Tips</div>
        <ul class="tips-list" aria-label="Preventive tips">${tipsHTML}</ul>
      ` : ''}
    </div>

    <button class="reset-btn" onclick="window.resetAnalysis()" aria-label="Start a new analysis">
      ↺ Analyze Another Leaf
    </button>
  `;
}

/* ── Build Feedback Card ──────────────────────────────────────── */
function buildFeedbackCard(predictionId, predictedLabel) {
  const labels = ['nitrogen', 'phosphorus', 'potassium', 'healthy'];
  const icons  = { nitrogen: '🟡', phosphorus: '🟣', potassium: '🟠', healthy: '🟢' };

  const btnHTML = labels.map(label => {
    const isCorrect = label === predictedLabel;
    return `
      <button
        class="feedback-btn${isCorrect ? ' feedback-correct' : ''}"
        data-label="${label}"
        aria-pressed="false"
        id="fb-btn-${label}">
        ${icons[label]} ${label.charAt(0).toUpperCase() + label.slice(1)}
        ${isCorrect ? '<span class="fb-predicted-tag">AI Prediction</span>' : ''}
      </button>`;
  }).join('');

  return `
    <div class="result-card feedback-card" id="feedback-card" data-prediction-id="${predictionId}" data-predicted-label="${predictedLabel}">
      <div class="card-title"><span>🧠</span> Help CropMind Learn</div>
      <p class="feedback-desc">Was this diagnosis correct? Your feedback improves future accuracy.</p>
      <div class="feedback-btns" role="group" aria-label="Feedback label selection">
        ${btnHTML}
      </div>
      <div class="feedback-submit-row" style="display:none;" id="feedback-submit-row">
        <button class="feedback-submit-btn" id="feedback-submit-btn">Submit Feedback ↗</button>
        <span class="feedback-selected-label" id="feedback-selected-label"></span>
      </div>
      <div class="feedback-done" id="feedback-done" style="display:none;">
        ✅ Thank you! Your feedback has been recorded and will improve the model.
      </div>
    </div>`;
}

/* ── Wire feedback card interactions ─────────────────────────── */
function wireFeedbackCard(panel) {
  const card = panel.querySelector('#feedback-card');
  if (!card) return;

  const predictionId  = card.dataset.predictionId  || '';
  const predictedLabel = card.dataset.predictedLabel || '';
  let   selectedLabel  = '';

  card.querySelectorAll('.feedback-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      card.querySelectorAll('.feedback-btn').forEach(b => {
        b.classList.remove('feedback-selected');
        b.setAttribute('aria-pressed', 'false');
      });
      this.classList.add('feedback-selected');
      this.setAttribute('aria-pressed', 'true');
      selectedLabel = this.dataset.label;

      const submitRow = document.getElementById('feedback-submit-row');
      const labelEl   = document.getElementById('feedback-selected-label');
      if (submitRow) submitRow.style.display = 'flex';
      if (labelEl)   labelEl.textContent = `Selected: ${selectedLabel}`;
    });
  });

  const submitBtn = document.getElementById('feedback-submit-btn');
  submitBtn?.addEventListener('click', async () => {
    if (!selectedLabel) return;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting…';

    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prediction_id:   predictionId,
          correct_label:   selectedLabel,
          predicted_label: predictedLabel,
        }),
      });
      const json = await res.json();
      if (json.success) {
        document.getElementById('feedback-done').style.display = 'block';
        document.getElementById('feedback-submit-row').style.display = 'none';
        card.querySelectorAll('.feedback-btn').forEach(b => b.disabled = true);

        const total = json.feedback_stats?.total || 0;
        const msg   = json.retrain_available
          ? `Feedback saved (${total} total). You can now retrain the model!`
          : `Feedback saved (${total} total). Keep going to enable retraining!`;
        window.showToast(msg, 'success', 5000);
      } else {
        window.showToast(json.error || 'Feedback submission failed.', 'error');
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit Feedback ↗';
      }
    } catch (err) {
      window.showToast('Network error. Please try again.', 'error');
      submitBtn.disabled = false;
      submitBtn.textContent = 'Submit Feedback ↗';
    }
  });
}

/* ── Main render function (called from upload.js) ─────────────── */
window.renderResults = function (data, originalSrc) {
  const panel = document.getElementById('results-panel');
  if (!panel) return;

  const predictionId   = data.prediction_id   || '';
  const predictedLabel = data.primary_deficiency || '';

  // Build all cards
  panel.innerHTML =
    buildDeficiencyCard(data) +
    buildHeatmapCard(data, originalSrc) +
    buildFertilizerCard(data) +
    buildInsightCard(data) +
    buildFeedbackCard(predictionId, predictedLabel);

  // ── Wire heatmap toggles (must happen after innerHTML is set) ──
  wireHeatmapToggles(panel);

  // ── Wire feedback card ────────────────────────────────────────
  wireFeedbackCard(panel);

  // ── Animate rings ─────────────────────────────────────────────
  const pctMap = { n: data.deficiencies.nitrogen, p: data.deficiencies.phosphorus, k: data.deficiencies.potassium };
  Object.entries(pctMap).forEach(([cls, raw], i) => {
    const pct = Math.round((raw || 0) * 100);
    const ringEl = panel.querySelector(`.ring-fill.${cls}`);
    const pctEl = panel.querySelector(`#ring-pct-${cls}`);
    if (ringEl) animateRing(ringEl, pct, i * 200 + 300);
    if (pctEl) {
      let count = 0;
      const target = pct;
      const dur = 1200;
      const step = Math.max(1, Math.round(target / (dur / 30)));
      const delay = i * 200 + 300;
      setTimeout(() => {
        const iv = setInterval(() => {
          count = Math.min(count + step, target);
          pctEl.textContent = `${count}%`;
          if (count >= target) clearInterval(iv);
        }, 30);
      }, delay);
    }
  });

  // ── Confidence bar ────────────────────────────────────────────
  const conf = Math.round((data.confidence || 0) * 100);
  setTimeout(() => {
    const confFill = document.getElementById('confidence-fill');
    const confVal  = document.getElementById('conf-value');
    if (confFill) confFill.style.width = `${conf}%`;
    if (confVal) {
      let c = 0;
      const iv = setInterval(() => {
        c = Math.min(c + 2, conf);
        confVal.textContent = `${c}%`;
        if (c >= conf) clearInterval(iv);
      }, 20);
    }
  }, 600);

  // ── Accordion toggles ─────────────────────────────────────────
  panel.querySelectorAll('.accordion-trigger').forEach(btn => {
    btn.addEventListener('click', function () {
      const isOpen = this.classList.toggle('open');
      this.setAttribute('aria-expanded', isOpen);
      const body = this.nextElementSibling;
      if (body) body.classList.toggle('open', isOpen);
    });
  });

  // ── Scroll to results ─────────────────────────────────────────
  setTimeout(() => {
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 400);

  window.showToast('Analysis complete! Review your results below.', 'success');
};
