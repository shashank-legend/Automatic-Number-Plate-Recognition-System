/**
 * dashboard.js
 * Frontend logic for the Confidence-Aware OCR Dashboard.
 * Novelty additions: Image Quality card, Plate DB lookup card,
 *   Closed-Loop Retry Trace, Session History drawer.
 */

'use strict';

// ── State ────────────────────────────────────────────────────────────────────
let currentFile  = null;
let allWordData  = [];
let filteredData = [];

// ── DOM references ────────────────────────────────────────────────────────────
const uploadZone    = document.getElementById('upload-zone');
const fileInput     = document.getElementById('file-input');
const uploadPreview = document.getElementById('upload-preview');
const previewName   = document.getElementById('preview-name');
const previewSize   = document.getElementById('preview-size');
const analyzeBtn    = document.getElementById('analyze-btn');
const changeBtn     = document.getElementById('change-btn');
const loadingPanel  = document.getElementById('loading-panel');
const resultsPanel  = document.getElementById('results-panel');
const resetBtn      = document.getElementById('reset-btn');
const wordSearch    = document.getElementById('word-search');
const confFilter    = document.getElementById('conf-filter');

// ── Upload zone events ────────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.querySelector('.upload-link').addEventListener('click', e => {
  e.stopPropagation();
  fileInput.click();
});

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

changeBtn.addEventListener('click', resetUpload);
resetBtn.addEventListener('click', resetAll);
analyzeBtn.addEventListener('click', runAnalysis);

// ── Crop canvas state ─────────────────────────────────────────────────────────
const cropCanvas  = document.getElementById('crop-canvas');
const cropCtx     = cropCanvas.getContext('2d');
const cropStatus  = document.getElementById('crop-status');
const clearCropBtn = document.getElementById('clear-crop-btn');

let naturalImg = new Image();   // full-resolution image
let selection  = null;          // {x, y, w, h} in CANVAS coordinates
let isDragging = false;
let dragStart  = { x: 0, y: 0 };

// ── File handling ─────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select an image file.');
    return;
  }
  currentFile = file;
  selection   = null;

  const reader = new FileReader();
  reader.onload = e => {
    naturalImg = new Image();
    naturalImg.onload = () => {
      drawCanvas();
      uploadZone.style.display    = 'none';
      uploadPreview.style.display = 'flex';
    };
    naturalImg.src = e.target.result;
  };
  reader.readAsDataURL(file);

  previewName.textContent = file.name;
  previewSize.textContent = formatBytes(file.size);
  clearSelection();
}

function drawCanvas() {
  const maxW = Math.min(700, cropCanvas.parentElement.clientWidth || 700);
  const scale = Math.min(1, maxW / naturalImg.naturalWidth);
  cropCanvas.width  = Math.round(naturalImg.naturalWidth  * scale);
  cropCanvas.height = Math.round(naturalImg.naturalHeight * scale);
  cropCtx.drawImage(naturalImg, 0, 0, cropCanvas.width, cropCanvas.height);
  if (selection) drawSelection();
}

function drawSelection() {
  const { x, y, w, h } = selection;
  cropCtx.fillStyle = 'rgba(0,0,0,0.45)';
  cropCtx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
  cropCtx.drawImage(naturalImg,
    x / cropCanvas.width  * naturalImg.naturalWidth,
    y / cropCanvas.height * naturalImg.naturalHeight,
    w / cropCanvas.width  * naturalImg.naturalWidth,
    h / cropCanvas.height * naturalImg.naturalHeight,
    x, y, w, h);
  cropCtx.strokeStyle = '#6366f1';
  cropCtx.lineWidth   = 2;
  cropCtx.setLineDash([6, 3]);
  cropCtx.strokeRect(x, y, w, h);
  cropCtx.setLineDash([]);
  [[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([cx, cy]) => {
    cropCtx.fillStyle = '#fff';
    cropCtx.fillRect(cx - 4, cy - 4, 8, 8);
    cropCtx.strokeStyle = '#6366f1';
    cropCtx.strokeRect(cx - 4, cy - 4, 8, 8);
  });
}

// ── Mouse events on canvas ────────────────────────────────────────────────────
cropCanvas.addEventListener('mousedown', e => {
  const r = cropCanvas.getBoundingClientRect();
  dragStart = { x: e.clientX - r.left, y: e.clientY - r.top };
  isDragging = true;
  selection  = null;
});

cropCanvas.addEventListener('mousemove', e => {
  if (!isDragging) return;
  const r = cropCanvas.getBoundingClientRect();
  const cx = e.clientX - r.left;
  const cy = e.clientY - r.top;
  selection = rectFromPoints(dragStart.x, dragStart.y, cx, cy);
  drawCanvas();
});

cropCanvas.addEventListener('mouseup', e => {
  if (!isDragging) return;
  isDragging = false;
  if (!selection || selection.w < 10 || selection.h < 10) {
    clearSelection();
  } else {
    const sw = Math.round(selection.w / cropCanvas.width  * naturalImg.naturalWidth);
    const sh = Math.round(selection.h / cropCanvas.height * naturalImg.naturalHeight);
    cropStatus.textContent = `✔ Selected ${sw} × ${sh} px — only this region will be analysed`;
    cropStatus.classList.add('has-selection');
    clearCropBtn.style.display = 'inline-flex';
  }
});

cropCanvas.addEventListener('mouseleave', () => { isDragging = false; });

// Touch support
cropCanvas.addEventListener('touchstart', e => {
  e.preventDefault();
  const t = e.touches[0], r = cropCanvas.getBoundingClientRect();
  dragStart = { x: t.clientX - r.left, y: t.clientY - r.top };
  isDragging = true; selection = null;
}, { passive: false });

cropCanvas.addEventListener('touchmove', e => {
  e.preventDefault();
  if (!isDragging) return;
  const t = e.touches[0], r = cropCanvas.getBoundingClientRect();
  selection = rectFromPoints(dragStart.x, dragStart.y, t.clientX - r.left, t.clientY - r.top);
  drawCanvas();
}, { passive: false });

cropCanvas.addEventListener('touchend', () => {
  isDragging = false;
  if (!selection || selection.w < 10 || selection.h < 10) { clearSelection(); }
  else {
    cropStatus.textContent = `✔ Selection active — only selected region will be analysed`;
    cropStatus.classList.add('has-selection');
    clearCropBtn.style.display = 'inline-flex';
  }
});

clearCropBtn.addEventListener('click', clearSelection);

function clearSelection() {
  selection = null;
  if (naturalImg.src) drawCanvas();
  cropStatus.textContent = 'No selection — full image will be used';
  cropStatus.classList.remove('has-selection');
  clearCropBtn.style.display = 'none';
}

function rectFromPoints(x1, y1, x2, y2) {
  return {
    x: Math.max(0, Math.min(x1, x2)),
    y: Math.max(0, Math.min(y1, y2)),
    w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1),
  };
}

// ── Get the image blob to send (cropped or full) ──────────────────────────────
function getImageBlob() {
  return new Promise(resolve => {
    if (!selection || selection.w < 10 || selection.h < 10) {
      resolve(currentFile);
      return;
    }
    const scaleX = naturalImg.naturalWidth  / cropCanvas.width;
    const scaleY = naturalImg.naturalHeight / cropCanvas.height;
    const nx = Math.round(selection.x * scaleX);
    const ny = Math.round(selection.y * scaleY);
    const nw = Math.round(selection.w * scaleX);
    const nh = Math.round(selection.h * scaleY);

    const offscreen = document.createElement('canvas');
    offscreen.width  = nw;
    offscreen.height = nh;
    offscreen.getContext('2d').drawImage(naturalImg, nx, ny, nw, nh, 0, 0, nw, nh);
    offscreen.toBlob(blob => resolve(blob), 'image/png');
  });
}

function formatBytes(bytes) {
  if (bytes < 1024)       return bytes + ' B';
  if (bytes < 1048576)    return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Run analysis ──────────────────────────────────────────────────────────────
async function runAnalysis() {
  if (!currentFile) return;

  uploadPreview.style.display = 'none';
  resultsPanel.style.display  = 'none';
  loadingPanel.style.display  = 'block';

  animateLoadingSteps();

  try {
    const blob     = await getImageBlob();
    const formData = new FormData();
    formData.append('image', blob, 'image.png');

    const resp = await fetch('/process', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!resp.ok || data.error) throw new Error(data.error || 'Server error');

    clearLoadingSteps();
    loadingPanel.style.display = 'none';
    populateResults(data);
    resultsPanel.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });

  } catch (err) {
    loadingPanel.style.display = 'none';
    uploadPreview.style.display = 'flex';
    alert('Error: ' + err.message);
  }
}

// ── Loading animation ─────────────────────────────────────────────────────────
let loadingInterval = null;
const loadingStepIds = ['ls1','ls2','ls3','ls4','ls5','ls6'];

function animateLoadingSteps() {
  let idx = 0;
  clearLoadingSteps();
  loadingInterval = setInterval(() => {
    if (idx < loadingStepIds.length) {
      const el = document.getElementById(loadingStepIds[idx]);
      if (el) el.classList.add('active');
      idx++;
    } else {
      clearInterval(loadingInterval);
    }
  }, 700);
}

function clearLoadingSteps() {
  if (loadingInterval) clearInterval(loadingInterval);
  loadingStepIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.classList.remove('active', 'done'); }
  });
}

// ── Populate results ──────────────────────────────────────────────────────────
function populateResults(data) {
  // Images
  document.getElementById('img-original').src     = 'data:image/png;base64,' + data.original_b64;
  document.getElementById('img-enhanced').src     = 'data:image/png;base64,' + data.enhanced_b64;
  document.getElementById('ann-original').src     = 'data:image/png;base64,' + data.annotated_original_b64;
  document.getElementById('ann-enhanced').src     = 'data:image/png;base64,' + data.annotated_enhanced_b64;

  // Stats cards
  buildStatsGrid(data.original_stats, data.enhanced_stats);

  // Confidence bars
  buildConfBar('bar-original', data.original_stats);
  buildConfBar('bar-enhanced', data.enhanced_stats);

  // ── Novelty 2: Image quality card ────────────────────────────────────────
  if (data.image_quality) buildQualityCard(data.image_quality);

  // ── Novelty 3: Retry trace ────────────────────────────────────────────────
  if (data.retry_trace && data.retry_trace.length > 0) {
    buildRetryTrace(data.retry_trace, data.plate_matched);
  }

  // ── Novelty: Plate display + DB lookup ───────────────────────────────────
  buildPlateLookup(data.plate_text, data.plate_matched, data.db_record);

  // ── Extracted text ────────────────────────────────────────────────────────
  const bestText  = data.enhanced_text  || '';
  const bestWords = data.enhanced_words || [];

  // Plate display
  const plateEl = document.getElementById('plate-display');
  if (data.plate_text && data.plate_text.length >= 6) {
    plateEl.textContent = data.plate_text;
    plateEl.className   = 'plate-display ' + (data.plate_matched ? 'valid' : 'invalid');
    plateEl.style.display = 'flex';
    plateEl.style.justifyContent = 'center';
    plateEl.style.alignItems = 'center';

    const sub = document.querySelector('.plate-sub') || (() => {
      const d = document.createElement('p'); d.className = 'plate-sub';
      plateEl.insertAdjacentElement('afterend', d); return d;
    })();
    sub.textContent = data.plate_matched
      ? '✓ Valid Indian number plate format'
      : '⚠ Format not recognised — may not be a plate image';
  } else {
    plateEl.style.display = 'none';
  }

  // Document-style display
  const docEl = document.getElementById('document-display');
  if (bestText.trim()) {
    docEl.textContent = bestText.trim();
    docEl.classList.remove('empty');
  } else {
    docEl.textContent = 'No text recognized';
    docEl.classList.add('empty');
  }

  document.getElementById('text-main').textContent = bestText || '(no text recognized)';

  // Engine badge
  const engineBadge = document.getElementById('ocr-engine-badge');
  if (data.ocr_engine) {
    engineBadge.textContent = '⚡ ' + data.ocr_engine;
    engineBadge.style.display = 'inline';
  } else {
    engineBadge.style.display = 'none';
  }

  // Copy button
  document.getElementById('copy-main-btn').onclick = () => {
    navigator.clipboard.writeText(bestText).then(() => {
      const btn = document.getElementById('copy-main-btn');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy text'; }, 1500);
    });
  };

  // Highlighted text
  document.getElementById('text-highlighted').innerHTML = buildHighlightedText(bestWords);

  // Word table
  allWordData  = bestWords;
  filteredData = [...allWordData];
  renderWordTable(filteredData);
}

// ── NOVELTY: Image Quality Card ───────────────────────────────────────────────
function buildQualityCard(iq) {
  const icons = { sharp: '🔆', motion: '💨', defocus: '🔅', mild_blur: '🌫' };
  const colors = { sharp: '#15803d', motion: '#b91c1c', defocus: '#a16207', mild_blur: '#1e40af' };
  const icon  = icons[iq.blur_type]  || '📷';
  const color = colors[iq.blur_type] || '#6366f1';

  const recPills = (iq.recommended_variants || [])
    .map(v => `<span class="rec-pill">${v}</span>`)
    .join('');

  document.getElementById('quality-content').innerHTML = `
    <div class="quality-type-row">
      <span class="quality-icon">${icon}</span>
      <div>
        <div class="quality-type-label" style="color:${color}">${capitalize(iq.blur_type.replace('_', ' '))}</div>
        <div class="quality-sub">${escapeHtml(iq.quality_label)}</div>
      </div>
    </div>
    <div class="quality-metrics">
      <div class="q-metric">
        <div class="q-metric-val">${iq.sharpness_score}</div>
        <div class="q-metric-label">Laplacian variance (sharpness)</div>
      </div>
      <div class="q-metric">
        <div class="q-metric-val">${(iq.directional_ratio * 100).toFixed(1)}%</div>
        <div class="q-metric-label">FFT directionality ratio</div>
      </div>
    </div>
    <div class="recommended-strip">
      <div class="rec-label">Recommended preprocessing variants for this blur type:</div>
      <div class="rec-variants">${recPills || '<span style="color:var(--muted);font-size:13px">all</span>'}</div>
    </div>
  `;
}

// ── NOVELTY: Plate DB Lookup Card ─────────────────────────────────────────────
function buildPlateLookup(plateText, plateMatched, dbRecord) {
  const container = document.getElementById('plate-db-content');
  const displayText = plateText && plateText.length >= 4 ? plateText : '—';
  const matchClass  = plateText ? (plateMatched ? 'matched' : 'unmatched') : '';

  let statusHtml = '';
  let recordHtml = '';

  if (!plateText || plateText.length < 4) {
    statusHtml = `<div class="db-status-banner notfound">ℹ No plate number extracted from this image.</div>`;
  } else if (dbRecord) {
    const statusClass = dbRecord.status === 'flagged' ? 'flagged'
                      : dbRecord.status === 'expired' ? 'expired'
                      : 'clear';
    const statusIcon  = dbRecord.status === 'flagged' ? '🚨'
                      : dbRecord.status === 'expired' ? '⚠️'
                      : '✅';
    const statusLabel = dbRecord.status === 'flagged' ? 'FLAGGED VEHICLE'
                      : dbRecord.status === 'expired' ? 'EXPIRED REGISTRATION'
                      : 'CLEAR — Vehicle found in registry';
    statusHtml = `<div class="db-status-banner ${statusClass}">${statusIcon} ${statusLabel}</div>`;
    recordHtml = `
      <div class="db-record-grid">
        <div class="db-rec-item"><div class="db-rec-key">Owner</div><div class="db-rec-val">${escapeHtml(dbRecord.owner_name || '—')}</div></div>
        <div class="db-rec-item"><div class="db-rec-key">Vehicle Type</div><div class="db-rec-val">${escapeHtml(dbRecord.vehicle_type || '—')}</div></div>
        <div class="db-rec-item"><div class="db-rec-key">Registered</div><div class="db-rec-val">${escapeHtml(dbRecord.registered || '—')}</div></div>
        <div class="db-rec-item"><div class="db-rec-key">Status</div><div class="db-rec-val">${escapeHtml(dbRecord.status || '—')}</div></div>
        ${dbRecord.note ? `<div class="db-rec-item" style="grid-column:1/-1"><div class="db-rec-key">Note</div><div class="db-rec-val" style="color:#b91c1c">${escapeHtml(dbRecord.note)}</div></div>` : ''}
      </div>
    `;
  } else {
    statusHtml = `<div class="db-status-banner notfound">🔎 Plate not found in local registry.</div>`;
  }

  container.innerHTML = `
    <div class="plate-lookup-number ${matchClass}">${escapeHtml(displayText)}</div>
    ${statusHtml}
    ${recordHtml}
  `;
}

// ── NOVELTY: Retry Trace ──────────────────────────────────────────────────────
function buildRetryTrace(trace, finalMatched) {
  const section = document.getElementById('retry-section');
  if (!trace || trace.length === 0) { section.style.display = 'none'; return; }

  // Only show if there were actual retries (more than just the initial attempt)
  const hadRetries = trace.length > 1 || (trace.length === 1 && !trace[0].matched);
  if (!hadRetries && trace[0] && trace[0].params && trace[0].params.includes('Direct match')) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';

  const rows = trace.map((t, i) => {
    const rowClass = t.matched ? 'retry-highlight' : '';
    const statusHtml = t.matched
      ? `<span class="retry-matched">✓ Valid</span>`
      : `<span class="retry-failed">✗ Retry</span>`;
    return `
      <tr class="${rowClass}">
        <td>${t.attempt}</td>
        <td style="font-size:12px;color:var(--muted)">${escapeHtml(t.params)}</td>
        <td style="font-family:'JetBrains Mono',monospace;font-weight:600">${escapeHtml(t.plate_text || '—')}</td>
        <td>${statusHtml}</td>
      </tr>`;
  }).join('');

  const summary = finalMatched
    ? `<p style="margin-top:10px;font-size:13px;color:#15803d;font-weight:500">✓ Valid Indian plate format found after ${trace.length} attempt(s).</p>`
    : `<p style="margin-top:10px;font-size:13px;color:#a16207">⚠ No exact format match found. Showing best result from ${trace.length} attempt(s).</p>`;

  document.getElementById('retry-trace').innerHTML = `
    <div class="table-wrap">
      <table class="retry-table">
        <thead><tr><th>#</th><th>Parameters Used</th><th>Extracted Text</th><th>Format Check</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${summary}
  `;
}

// ── Stats grid ────────────────────────────────────────────────────────────────
function buildStatsGrid(orig, enh) {
  const improvement = (enh.avg_confidence - orig.avg_confidence).toFixed(1);
  const improvClass = parseFloat(improvement) >= 0 ? 'color-green' : 'color-red';
  const improvSign  = parseFloat(improvement) >= 0 ? '+' : '';

  document.getElementById('stats-grid').innerHTML = `
    <div class="stat-card">
      <div class="stat-value color-blue">${orig.avg_confidence}%</div>
      <div class="stat-label">Original avg confidence</div>
      <div class="stat-sub">${orig.total_words} words</div>
    </div>
    <div class="stat-card">
      <div class="stat-value color-blue">${enh.avg_confidence}%</div>
      <div class="stat-label">Enhanced avg confidence</div>
      <div class="stat-sub">${enh.total_words} words</div>
    </div>
    <div class="stat-card improvement">
      <div class="stat-value ${improvClass}">${improvSign}${improvement}%</div>
      <div class="stat-label">Confidence improvement</div>
      <div class="stat-sub">after preprocessing</div>
    </div>
    <div class="stat-card">
      <div class="stat-value color-green">${enh.high_count}</div>
      <div class="stat-label">High confidence words</div>
      <div class="stat-sub">${enh.high_pct}% of total</div>
    </div>
    <div class="stat-card">
      <div class="stat-value color-yellow">${enh.medium_count}</div>
      <div class="stat-label">Medium confidence</div>
      <div class="stat-sub">${enh.medium_pct}% of total</div>
    </div>
    <div class="stat-card">
      <div class="stat-value color-red">${enh.low_count}</div>
      <div class="stat-label">Low confidence words</div>
      <div class="stat-sub">${enh.low_pct}% of total</div>
    </div>
  `;
}

// ── Confidence bar ────────────────────────────────────────────────────────────
function buildConfBar(containerId, stats) {
  document.getElementById(containerId).innerHTML = `
    <div class="conf-bar-row">
      <span class="conf-bar-row-label color-green">High</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill green" style="width:${stats.high_pct}%"></div>
      </div>
      <span class="conf-bar-pct">${stats.high_pct}%</span>
    </div>
    <div class="conf-bar-row">
      <span class="conf-bar-row-label color-yellow">Medium</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill yellow" style="width:${stats.medium_pct}%"></div>
      </div>
      <span class="conf-bar-pct">${stats.medium_pct}%</span>
    </div>
    <div class="conf-bar-row">
      <span class="conf-bar-row-label color-red">Low</span>
      <div class="conf-bar-track">
        <div class="conf-bar-fill red" style="width:${stats.low_pct}%"></div>
      </div>
      <span class="conf-bar-pct">${stats.low_pct}%</span>
    </div>
  `;
}

// ── Highlighted text ──────────────────────────────────────────────────────────
function buildHighlightedText(words) {
  if (!words || !words.length) return '<em>No text recognized</em>';
  return words.map((w, idx) => {
    const lvl = getLevel(w.confidence);
    return `<span class="h-word ${lvl}" 
                  title="Confidence: ${w.confidence}%" 
                  data-idx="${idx}"
                  onclick="highlightWordInTable(${idx})">${escapeHtml(w.word)}</span> `;
  }).join('');
}

window.highlightWordInTable = function(idx) {
  document.querySelectorAll('.h-word').forEach(el => el.classList.remove('active-box'));
  const wordEl = document.querySelector(`.h-word[data-idx="${idx}"]`);
  if (wordEl) wordEl.classList.add('active-box');
  const row = document.querySelector(`#word-tbody tr:nth-child(${idx + 1})`);
  if (row) {
    row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    row.style.background = '#eff6ff';
    setTimeout(() => { row.style.background = ''; }, 2000);
  }
};

// ── Word table ────────────────────────────────────────────────────────────────
function renderWordTable(words) {
  const tbody = document.getElementById('word-tbody');
  if (!words.length) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#9ca3af;padding:24px">No words match the filter</td></tr>';
    return;
  }
  tbody.innerHTML = words.map((w, i) => {
    const lvl = getLevel(w.confidence);
    return `
      <tr>
        <td>${i + 1}</td>
        <td><strong>${escapeHtml(w.word)}</strong></td>
        <td><span class="conf-num ${lvl}">${w.confidence}%</span></td>
        <td><span class="pill ${lvl}">${lvl.charAt(0).toUpperCase() + lvl.slice(1)}</span></td>
        <td style="color:#9ca3af;font-size:12px">(${w.x}, ${w.y})</td>
      </tr>`;
  }).join('');
}

// ── Table filters ─────────────────────────────────────────────────────────────
wordSearch.addEventListener('input', applyFilters);
confFilter.addEventListener('change', applyFilters);

function applyFilters() {
  const q   = wordSearch.value.toLowerCase().trim();
  const lvl = confFilter.value;
  filteredData = allWordData.filter(w => {
    const matchText = !q || w.word.toLowerCase().includes(q);
    const matchLvl  = lvl === 'all' || getLevel(w.confidence) === lvl;
    return matchText && matchLvl;
  });
  renderWordTable(filteredData);
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab[data-tab]').forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    document.querySelectorAll('.tab[data-tab]').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-' + target).classList.add('active');
  });
});

// ── History Drawer ────────────────────────────────────────────────────────────
const historyToggleBtn = document.getElementById('history-toggle-btn');
const historyDrawer    = document.getElementById('history-drawer');
const historyOverlay   = document.getElementById('history-overlay');

historyToggleBtn.addEventListener('click', openHistory);

async function openHistory() {
  historyOverlay.style.display = 'block';
  historyDrawer.classList.add('open');
  try {
    const resp = await fetch('/history');
    const data = await resp.json();
    renderHistory(data.history || []);
  } catch (e) {
    document.getElementById('history-list').innerHTML =
      '<p class="history-empty">Could not load history.</p>';
  }
}

window.closeHistory = function() {
  historyDrawer.classList.remove('open');
  setTimeout(() => { historyOverlay.style.display = 'none'; }, 300);
};

function renderHistory(rows) {
  const el = document.getElementById('history-list');
  if (!rows.length) {
    el.innerHTML = '<p class="history-empty">No scans yet. Analyze an image to see history here.</p>';
    return;
  }
  el.innerHTML = rows.map(r => {
    const ts = r.timestamp ? r.timestamp.replace('T', ' ').slice(0, 19) + ' UTC' : '—';
    const matchTag = r.plate_matched
      ? '<span class="htag htag-green">✓ Valid plate</span>'
      : '<span class="htag htag-yellow">Unmatched</span>';
    const blurTag = r.blur_type
      ? `<span class="htag htag-blue">${r.blur_type}</span>` : '';
    const confTag = r.avg_confidence >= 70
      ? `<span class="htag htag-green">${r.avg_confidence.toFixed(0)}% conf</span>`
      : `<span class="htag htag-yellow">${r.avg_confidence.toFixed(0)}% conf</span>`;

    return `
      <div class="history-item">
        <div class="history-plate">${escapeHtml(r.plate_text || '—')}</div>
        <div class="history-meta">${escapeHtml(r.filename || '')} · ${ts}</div>
        <div class="history-tags">${matchTag}${blurTag}${confTag}</div>
      </div>`;
  }).join('');
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetUpload() {
  currentFile = null;
  fileInput.value = '';
  clearSelection();
  cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
  uploadPreview.style.display = 'none';
  uploadZone.style.display    = 'block';
}

function resetAll() {
  resetUpload();
  resultsPanel.style.display = 'none';
  allWordData  = [];
  filteredData = [];
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function getLevel(conf) {
  if (conf >= 80) return 'high';
  if (conf >= 50) return 'medium';
  return 'low';
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}
