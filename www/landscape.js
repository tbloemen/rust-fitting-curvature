// GP Landscape visualization
// Loads results/results.jsonl, fits a GP for the selected (dataset, curvature, metric),
// and renders a 2D posterior-mean heatmap for any two chosen hyperparameters.

const PARAMS = [
  { name: "learning_rate",      label: "Learning Rate",      log: true,  min: 0.5,    max: 50.0  },
  { name: "perplexity_ratio",   label: "Perplexity Ratio",   log: true,  min: 4e-4,   max: 0.03  },
  { name: "momentum_main",      label: "Momentum",           log: false, min: 0.60,   max: 1.0   },
  { name: "centering_weight",   label: "Centering Weight",   log: false, min: 0.0,    max: 2.0   },
  { name: "global_loss_weight", label: "Global Loss Weight", log: false, min: 0.0,    max: 2.0   },
  { name: "norm_loss_weight",   label: "Norm Loss Weight",   log: false, min: 0.0,    max: 0.02  },
];

const METRICS = [
  { key: "trustworthiness",            label: "Trustworthiness",          maximize: true  },
  { key: "continuity",                 label: "Continuity",               maximize: true  },
  { key: "knn_overlap",                label: "KNN Overlap",              maximize: true  },
  { key: "geodesic_distortion_gu2019", label: "Geodesic Distortion GU19", maximize: false },
  { key: "geodesic_distortion_mse",    label: "Geodesic Distortion MSE",  maximize: false },
  { key: "davies_bouldin_ratio",       label: "Davies-Bouldin Ratio",     maximize: true  },
  { key: "dunn_index",                 label: "Dunn Index",               maximize: true  },
  { key: "class_density_measure",      label: "Class Density Measure",    maximize: true  },
  { key: "cluster_density_measure",    label: "Cluster Density Measure",  maximize: true  },
];

// Viridis colormap — 12-point LUT, linearly interpolated
const VIRIDIS_LUT = [
  [68, 1, 84], [72, 33, 115], [67, 62, 133], [56, 88, 140],
  [45, 112, 142], [37, 133, 142], [30, 155, 138], [42, 176, 127],
  [82, 197, 105], [134, 213, 73], [194, 223, 35], [253, 231, 37],
];

function viridisRgb(t) {
  t = Math.max(0, Math.min(1, t));
  const n = VIRIDIS_LUT.length - 1;
  const f = t * n;
  const lo = Math.floor(f), hi = Math.min(lo + 1, n), frac = f - lo;
  const [r0, g0, b0] = VIRIDIS_LUT[lo], [r1, g1, b1] = VIRIDIS_LUT[hi];
  return [
    Math.round(r0 + (r1 - r0) * frac),
    Math.round(g0 + (g1 - g0) * frac),
    Math.round(b0 + (b1 - b0) * frac),
  ];
}

// ===== Math helpers =====

function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }
function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }
function std(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

// ===== GP kernel and solver =====

function rbfKernel(a, b, ls) {
  let sq = 0;
  for (let i = 0; i < a.length; i++) sq += (a[i] - b[i]) ** 2;
  return Math.exp(-sq / (2 * ls * ls));
}

// Cholesky decomposition: returns lower-triangular L such that A ≈ L * L^T
function choleskyDecomp(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
    }
  }
  return L;
}

// Solve A*x = b where A = L*L^T (Cholesky factors given)
function solveChol(L, b) {
  const n = L.length;
  // Forward: L * y = b
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let j = 0; j < i; j++) s -= L[i][j] * y[j];
    y[i] = s / L[i][i];
  }
  // Backward: L^T * x = y
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= L[j][i] * x[j];
    x[i] = s / L[i][i];
  }
  return x;
}

// ===== Data loading =====

async function loadResults() {
  const resp = await fetch("/results/results.jsonl");
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching results.jsonl`);
  const text = await resp.text();
  return text.trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));
}

// Try to load a pre-fitted GP state file for the given dataset+curvature.
// Returns the parsed JSON or null if not found.
async function loadGpState(dataset, curvature) {
  const sign = curvature >= 0 ? "+" : "-";
  const absStr = Math.abs(curvature).toFixed(1).replace(".", "_");
  const name = `results_gp_${dataset}_${sign}${absStr}.json`;
  try {
    const resp = await fetch(`/results/${name}`);
    if (resp.ok) return await resp.json();
  } catch (_) { /* ignore */ }
  return null;
}

// ===== Normalization =====

// Encode a single parameter value (log-transform if needed)
function encode(value, param) {
  return param.log ? Math.log(Math.max(+value, 1e-15)) : +value;
}

// Encode a full result row into a 6-element array (one entry per PARAMS element)
function encodeRow(row) {
  return PARAMS.map((p) => encode(row[p.name], p));
}

// Compute per-dimension mean and std of encoded inputs over a set of rows
function computeNormStats(rows) {
  const encoded = rows.map(encodeRow);
  return {
    xMeans: PARAMS.map((_, i) => mean(encoded.map((e) => e[i]))),
    xStds: PARAMS.map((_, i) => {
      const s = std(encoded.map((e) => e[i]));
      return s < 1e-12 ? 1 : s;
    }),
  };
}

function normalizeVec(encoded, xMeans, xStds) {
  return encoded.map((v, i) => (v - xMeans[i]) / xStds[i]);
}

// ===== GP model =====

// Build GP model from a set of result rows for a specific metric.
// normStats: { xMeans, xStds } — input standardization
// lengthScale: RBF kernel length scale (in normalized space)
function buildGpModel(rows, metricKey, normStats, lengthScale) {
  const valid = rows.filter((r) => r[metricKey] != null && isFinite(+r[metricKey]));
  if (valid.length < 2) return null;

  const xNorm = valid.map((r) =>
    normalizeVec(encodeRow(r), normStats.xMeans, normStats.xStds)
  );
  const yRaw = valid.map((r) => +r[metricKey]);

  const yMean = mean(yRaw);
  const yStdRaw = std(yRaw);
  const yStd = yStdRaw < 1e-12 ? 1 : yStdRaw;
  const yNorm = yRaw.map((v) => (v - yMean) / yStd);

  // K(X, X) + jitter
  const n = valid.length;
  const K = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) => rbfKernel(xNorm[i], xNorm[j], lengthScale))
  );
  for (let i = 0; i < n; i++) K[i][i] += 1e-4;

  const L = choleskyDecomp(K);
  const alpha = Array.from(solveChol(L, yNorm));

  // Best observation (used to fix the unselected axes when slicing to 2D)
  const metricInfo = METRICS.find((m) => m.key === metricKey);
  const bestObs = valid.reduce((b, o) =>
    (metricInfo.maximize ? +o[metricKey] > +b[metricKey] : +o[metricKey] < +b[metricKey]) ? o : b
  );

  return { alpha, xNorm, yMean, yStd, lengthScale, observations: valid, normStats, bestObs };
}

// Evaluate GP posterior mean at a normalized test point
function predictGP(model, xTestNorm) {
  const kStar = model.xNorm.map((xi) => rbfKernel(xi, xTestNorm, model.lengthScale));
  return dot(kStar, model.alpha) * model.yStd + model.yMean;
}

// Predict at a grid point (xParamIdx=xv, yParamIdx=yv, others fixed at bestObs)
function predictAtGrid(model, xParamIdx, xv, yParamIdx, yv) {
  const tempRow = { ...model.bestObs };
  tempRow[PARAMS[xParamIdx].name] = xv;
  tempRow[PARAMS[yParamIdx].name] = yv;
  const norm = normalizeVec(
    encodeRow(tempRow),
    model.normStats.xMeans,
    model.normStats.xStds
  );
  return predictGP(model, norm);
}

// ===== Grid helpers =====

const GRID_N = 50;

function linspace(lo, hi, n, log = false) {
  if (log) {
    const a = Math.log(lo), b = Math.log(hi);
    return Array.from({ length: n }, (_, i) => Math.exp(a + (b - a) * i / (n - 1)));
  }
  return Array.from({ length: n }, (_, i) => lo + (hi - lo) * i / (n - 1));
}

// Convert a natural-space value to a fractional position in [0, 1] within [lo, hi]
function valueToFrac(v, lo, hi, log) {
  if (log) {
    const lv = Math.log(Math.max(v, 1e-15));
    return (lv - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
  }
  return (v - lo) / (hi - lo);
}

function formatTick(v, log) {
  const a = Math.abs(v);
  if (log) {
    if (a < 1e-3) return v.toExponential(0);
    if (a < 1e-2) return v.toFixed(4);
    if (a < 0.1)  return v.toFixed(3);
    if (a < 1)    return v.toFixed(2);
    if (a < 10)   return v.toFixed(1);
    return v.toFixed(0);
  }
  if (v === 0) return "0";
  if (a < 1e-3) return v.toExponential(1);
  if (a < 1e-2) return v.toFixed(4);
  if (a < 0.1)  return v.toFixed(3);
  if (a < 1)    return v.toFixed(2);
  if (a < 10)   return v.toFixed(1);
  return v.toFixed(0);
}

// ===== Canvas rendering =====

// Pixel margins reserved for axis labels (must match colorbar-wrapper padding in CSS)
const MARGIN = { left: 72, right: 12, top: 15, bottom: 54 };

function renderHeatmap(canvas, model, xParamIdx, yParamIdx, predictions, xVals, yVals, vmin, vmax) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const plotW = W - MARGIN.left - MARGIN.right;
  const plotH = H - MARGIN.top - MARGIN.bottom;

  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, W, H);

  if (plotW <= 0 || plotH <= 0) return;

  const xParam = PARAMS[xParamIdx], yParam = PARAMS[yParamIdx];
  const range = vmax - vmin;

  // Draw heatmap using ImageData (much faster than per-cell fillRect)
  const imgData = ctx.createImageData(plotW, plotH);
  const buf = imgData.data;
  for (let xi = 0; xi < GRID_N; xi++) {
    const px0 = Math.round(xi * plotW / GRID_N);
    const px1 = Math.round((xi + 1) * plotW / GRID_N);
    for (let yi = 0; yi < GRID_N; yi++) {
      const t = range > 1e-12 ? (predictions[xi * GRID_N + yi] - vmin) / range : 0.5;
      const [r, g, b] = viridisRgb(t);
      // yi=0 is the bottom of the plot, so flip vertically
      const py0 = Math.round((GRID_N - 1 - yi) * plotH / GRID_N);
      const py1 = Math.round((GRID_N - yi) * plotH / GRID_N);
      for (let px = px0; px < px1; px++) {
        for (let py = py0; py < py1; py++) {
          const idx = (py * plotW + px) * 4;
          buf[idx] = r; buf[idx + 1] = g; buf[idx + 2] = b; buf[idx + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(imgData, MARGIN.left, MARGIN.top);

  // Plot border
  ctx.strokeStyle = "#555";
  ctx.lineWidth = 1;
  ctx.strokeRect(MARGIN.left + 0.5, MARGIN.top + 0.5, plotW, plotH);

  // X axis ticks and label
  ctx.fillStyle = "#aaa";
  ctx.font = "11px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const xTicks = linspace(xParam.min, xParam.max, 5, xParam.log);
  for (const v of xTicks) {
    const fx = valueToFrac(v, xParam.min, xParam.max, xParam.log);
    const px = MARGIN.left + fx * plotW;
    ctx.fillStyle = "#666";
    ctx.fillRect(px - 0.5, MARGIN.top + plotH, 1, 5);
    ctx.fillStyle = "#aaa";
    ctx.fillText(formatTick(v, xParam.log), px, MARGIN.top + plotH + 8);
  }
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#ccc";
  ctx.textBaseline = "bottom";
  ctx.fillText(xParam.label, MARGIN.left + plotW / 2, H - 2);

  // Y axis ticks and label
  ctx.font = "11px monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const yTicks = linspace(yParam.min, yParam.max, 5, yParam.log);
  for (const v of yTicks) {
    const fy = valueToFrac(v, yParam.min, yParam.max, yParam.log);
    const py = MARGIN.top + plotH - fy * plotH;
    ctx.fillStyle = "#666";
    ctx.fillRect(MARGIN.left - 5, py - 0.5, 5, 1);
    ctx.fillStyle = "#aaa";
    ctx.fillText(formatTick(v, yParam.log), MARGIN.left - 8, py);
  }
  ctx.save();
  ctx.translate(13, MARGIN.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#ccc";
  ctx.fillText(yParam.label, 0, 0);
  ctx.restore();

  // Observation scatter (white dots with dark shadow)
  for (const obs of model.observations) {
    const xv = +obs[xParam.name], yv = +obs[yParam.name];
    if (!isFinite(xv) || !isFinite(yv)) continue;
    const fx = valueToFrac(xv, xParam.min, xParam.max, xParam.log);
    const fy = valueToFrac(yv, yParam.min, yParam.max, yParam.log);
    const px = MARGIN.left + Math.max(0, Math.min(1, fx)) * plotW;
    const py = MARGIN.top + (1 - Math.max(0, Math.min(1, fy))) * plotH;

    ctx.beginPath();
    ctx.arc(px, py, 5.5, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(px, py, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "white";
    ctx.fill();
  }
}

function renderColorbar(canvas, vmin, vmax) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  if (H <= 0) return;
  const imgData = ctx.createImageData(W, H);
  const buf = imgData.data;
  for (let py = 0; py < H; py++) {
    const t = 1 - py / (H - 1);
    const [r, g, b] = viridisRgb(t);
    for (let px = 0; px < W; px++) {
      const idx = (py * W + px) * 4;
      buf[idx] = r; buf[idx + 1] = g; buf[idx + 2] = b; buf[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

// ===== App =====

let allResults = [];
let lastRenderState = null;

async function init() {
  setStatus("Loading results…");
  try {
    allResults = await loadResults();
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    return;
  }

  populateDatasets();
  populateParams();
  populateMetrics();

  document.getElementById("ld-dataset").addEventListener("change", populateCurvatures);
  populateCurvatures();

  document.getElementById("ld-render-btn").addEventListener("click", onRender);
  document.getElementById("ld-render-btn").disabled = false;

  // Resize canvas whenever its wrapper changes size
  const wrapper = document.querySelector(".heatmap-wrapper");
  const ro = new ResizeObserver(() => {
    syncCanvasSize();
    if (lastRenderState) drawAll(lastRenderState);
  });
  ro.observe(wrapper);
  syncCanvasSize();

  setStatus(`${allResults.length} results loaded`);
}

function populateDatasets() {
  const datasets = [...new Set(allResults.map((r) => r.dataset_name))].sort();
  const sel = document.getElementById("ld-dataset");
  sel.innerHTML = datasets.map((d) => `<option value="${d}">${d}</option>`).join("");
}

function populateCurvatures() {
  const dataset = document.getElementById("ld-dataset").value;
  const curvatures = [
    ...new Set(allResults.filter((r) => r.dataset_name === dataset).map((r) => r.curvature)),
  ].sort((a, b) => a - b);
  const sel = document.getElementById("ld-curvature");
  sel.innerHTML = curvatures.map((c) => `<option value="${c}">${c}</option>`).join("");
}

function populateParams() {
  const xSel = document.getElementById("ld-x-param");
  const ySel = document.getElementById("ld-y-param");
  const opts = PARAMS.map((p, i) => `<option value="${i}">${p.label}</option>`).join("");
  xSel.innerHTML = ySel.innerHTML = opts;
  xSel.value = "0"; // learning_rate
  ySel.value = "1"; // perplexity_ratio
}

function populateMetrics() {
  const sel = document.getElementById("ld-metric");
  sel.innerHTML = METRICS.map((m) => `<option value="${m.key}">${m.label}</option>`).join("");
}

function syncCanvasSize() {
  const wrapper = document.querySelector(".heatmap-wrapper");
  const canvas = document.getElementById("heatmap");
  const r = wrapper.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(r.width));
  canvas.height = Math.max(1, Math.floor(r.height));
}

async function onRender() {
  const dataset   = document.getElementById("ld-dataset").value;
  const curvature = parseFloat(document.getElementById("ld-curvature").value);
  const xParamIdx = parseInt(document.getElementById("ld-x-param").value);
  const yParamIdx = parseInt(document.getElementById("ld-y-param").value);
  const metricKey = document.getElementById("ld-metric").value;

  if (xParamIdx === yParamIdx) {
    setStatus("X and Y axes must be different parameters.");
    return;
  }

  const obs = allResults.filter(
    (r) => r.dataset_name === dataset && r.curvature === curvature
  );
  const valid = obs.filter((r) => r[metricKey] != null && isFinite(+r[metricKey]));
  if (valid.length < 2) {
    setStatus(`Only ${valid.length} valid observation(s) for this metric — need ≥ 2.`);
    return;
  }

  document.getElementById("ld-render-btn").disabled = true;
  setStatus("Computing…");

  // Use GP state file's length_scale when available (better estimate than the default 1.0)
  let lengthScale = 1.0;
  const gpState = await loadGpState(dataset, curvature);
  if (gpState && typeof gpState.length_scale === "number") {
    lengthScale = Math.max(gpState.length_scale, 0.01);
  }

  // Always compute normalization from the current valid observations
  const normStats = computeNormStats(valid);
  const model = buildGpModel(valid, metricKey, normStats, lengthScale);
  if (!model) {
    setStatus("Failed to build GP model.");
    document.getElementById("ld-render-btn").disabled = false;
    return;
  }

  // Evaluate GP on GRID_N × GRID_N grid
  const xParam = PARAMS[xParamIdx], yParam = PARAMS[yParamIdx];
  const xVals = linspace(xParam.min, xParam.max, GRID_N, xParam.log);
  const yVals = linspace(yParam.min, yParam.max, GRID_N, yParam.log);

  const predictions = new Float64Array(GRID_N * GRID_N);
  for (let xi = 0; xi < GRID_N; xi++) {
    for (let yi = 0; yi < GRID_N; yi++) {
      predictions[xi * GRID_N + yi] = predictAtGrid(
        model, xParamIdx, xVals[xi], yParamIdx, yVals[yi]
      );
    }
  }

  const vmin = Math.min(...predictions), vmax = Math.max(...predictions);
  const metricInfo = METRICS.find((m) => m.key === metricKey);

  lastRenderState = {
    model, xParamIdx, yParamIdx, predictions, xVals, yVals, vmin, vmax, metricInfo,
  };
  drawAll(lastRenderState);

  document.getElementById("ld-render-btn").disabled = false;
  const lsSource = gpState ? "GP state" : "default";
  setStatus(
    `${valid.length} obs · ls=${lengthScale.toFixed(3)} (${lsSource}) · k=${curvature}\n` +
    `Other params fixed at best: ${PARAMS.map(p => p.name).filter((_, i) => i !== xParamIdx && i !== yParamIdx).map(n => `${n}=${(+model.bestObs[n]).toFixed(3)}`).join(", ")}`
  );
}

function drawAll({ model, xParamIdx, yParamIdx, predictions, xVals, yVals, vmin, vmax }) {
  syncCanvasSize();
  const canvas = document.getElementById("heatmap");
  renderHeatmap(canvas, model, xParamIdx, yParamIdx, predictions, xVals, yVals, vmin, vmax);

  // Match colorbar height to the actual plot area height
  const cbCanvas = document.getElementById("colorbar");
  cbCanvas.height = Math.max(1, canvas.height - MARGIN.top - MARGIN.bottom);
  renderColorbar(cbCanvas, vmin, vmax);

  document.getElementById("cb-max").textContent = vmax.toFixed(4);
  document.getElementById("cb-min").textContent = vmin.toFixed(4);
}

function setStatus(msg) {
  document.getElementById("ld-status").textContent = msg;
}

init();
