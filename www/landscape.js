// GP Landscape visualization
// Loads results/results.jsonl and renders a 2D GP posterior-mean heatmap.
//
// Two modes:
//   single     — GP on one chosen metric, colorbar in metric units.
//   chebyshev  — GP on augmented-Chebyshev scalarization of the 10 Pareto
//                objectives (equal weights).  Yellow = closer to ideal.
//
// In both modes, Pareto-front configs (from *_pareto_*.json) are drawn as
// gold circles on top of the regular observation scatter.

const PARAMS = [
  {
    name: "learning_rate",
    label: "Learning Rate",
    log: true,
    min: 0.5,
    max: 50.0,
  },
  {
    name: "perplexity_ratio",
    label: "Perplexity Ratio",
    log: true,
    min: 4e-4,
    max: 0.03,
  },
  { name: "momentum_main", label: "Momentum", log: false, min: 0.6, max: 1.0 },
  {
    name: "centering_weight",
    label: "Centering Weight",
    log: false,
    min: 0.0,
    max: 2.0,
  },
  {
    name: "global_loss_weight",
    label: "Global Loss Weight",
    log: false,
    min: 0.0,
    max: 2.0,
  },
  {
    name: "norm_loss_weight",
    label: "Norm Loss Weight",
    log: false,
    min: 0.0,
    max: 0.02,
  },
  {
    name: "curvature_magnitude",
    label: "Curvature Magnitude",
    log: true,
    min: 0.01,
    max: 25.0,
  },
];

const METRICS = [
  { key: "trustworthiness", label: "Trustworthiness", maximize: true },
  {
    key: "trustworthiness_manifold",
    label: "Trustworthiness (manifold)",
    maximize: true,
  },
  { key: "continuity", label: "Continuity", maximize: true },
  {
    key: "continuity_manifold",
    label: "Continuity (manifold)",
    maximize: true,
  },
  { key: "knn_overlap", label: "KNN Overlap", maximize: true },
  {
    key: "knn_overlap_manifold",
    label: "KNN Overlap (manifold)",
    maximize: true,
  },
  { key: "neighborhood_hit", label: "Neighborhood Hit", maximize: true },
  {
    key: "neighborhood_hit_manifold",
    label: "Neighborhood Hit (manifold)",
    maximize: true,
  },
  { key: "normalized_stress", label: "Normalized Stress", maximize: false },
  {
    key: "normalized_stress_manifold",
    label: "Normalized Stress (manifold)",
    maximize: false,
  },
  { key: "shepard_goodness", label: "Shepard Goodness", maximize: true },
  {
    key: "shepard_goodness_manifold",
    label: "Shepard Goodness (manifold)",
    maximize: true,
  },
  {
    key: "davies_bouldin_ratio",
    label: "Davies-Bouldin Ratio",
    maximize: true,
  },
  { key: "dunn_index", label: "Dunn Index", maximize: true },
  {
    key: "class_density_measure",
    label: "Class Density Measure",
    maximize: true,
  },
  {
    key: "cluster_density_measure",
    label: "Cluster Density Measure",
    maximize: true,
  },
];

// The 10 objectives used in --mode pareto (matches default_pareto_metrics() in main.rs).
const PARETO_OBJECTIVES = [
  { key: "trustworthiness", maximize: true },
  { key: "trustworthiness_manifold", maximize: true },
  { key: "continuity", maximize: true },
  { key: "continuity_manifold", maximize: true },
  { key: "normalized_stress", maximize: false },
  { key: "normalized_stress_manifold", maximize: false },
  { key: "shepard_goodness", maximize: true },
  { key: "shepard_goodness_manifold", maximize: true },
  { key: "neighborhood_hit", maximize: true },
  { key: "neighborhood_hit_manifold", maximize: true },
];

// Viridis colormap — 12-point LUT, linearly interpolated
const VIRIDIS_LUT = [
  [68, 1, 84],
  [72, 33, 115],
  [67, 62, 133],
  [56, 88, 140],
  [45, 112, 142],
  [37, 133, 142],
  [30, 155, 138],
  [42, 176, 127],
  [82, 197, 105],
  [134, 213, 73],
  [194, 223, 35],
  [253, 231, 37],
];

function viridisRgb(t) {
  t = Math.max(0, Math.min(1, t));
  const n = VIRIDIS_LUT.length - 1;
  const f = t * n;
  const lo = Math.floor(f),
    hi = Math.min(lo + 1, n),
    frac = f - lo;
  const [r0, g0, b0] = VIRIDIS_LUT[lo],
    [r1, g1, b1] = VIRIDIS_LUT[hi];
  return [
    Math.round(r0 + (r1 - r0) * frac),
    Math.round(g0 + (g1 - g0) * frac),
    Math.round(b0 + (b1 - b0) * frac),
  ];
}

// ===== Math helpers =====

function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}
function mean(arr) {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}
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
  const file =
    new URLSearchParams(window.location.search).get("file") ?? "results.jsonl";
  const resp = await fetch(`/results/${file}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${file}`);
  const text = await resp.text();
  return text
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

// Try to load a pre-fitted GP state file keyed by geometry string.
// e.g. results_gp_mnist_hyperbolic.json
async function loadGpState(dataset, geometry) {
  try {
    const resp = await fetch(`/results/results_gp_${dataset}_${geometry}.json`);
    if (resp.ok) return await resp.json();
  } catch (_) {}
  return null;
}

// Load Pareto front JSON written by --mode pareto.
// Returns an array of entries, each with config fields at top level and a
// `metrics` sub-object.  Returns null if the file doesn't exist.
async function loadParetoFront(dataset, geometry) {
  try {
    const resp = await fetch(
      `/results/results_pareto_${dataset}_${geometry}.json`,
    );
    if (resp.ok) return await resp.json();
  } catch (_) {}
  return null;
}

// ===== Normalization =====

// Encode a single parameter value (log-transform if needed)
function encode(value, param) {
  return param.log ? Math.log(Math.max(+value, 1e-15)) : +value;
}

// Determine which PARAMS are present (have finite values) in a set of rows.
function activeParams(rows) {
  return PARAMS.filter((p) =>
    rows.some(
      (r) => r[p.name] != null && isFinite(+r[p.name]) && +r[p.name] !== 0,
    ),
  );
}

// Encode a full result row into an array (one entry per active param element)
function encodeRow(row, params) {
  return params.map((p) => encode(row[p.name], p));
}

// Compute per-dimension mean and std of encoded inputs over a set of rows
function computeNormStats(rows, params) {
  const encoded = rows.map((r) => encodeRow(r, params));
  return {
    xMeans: params.map((_, i) => mean(encoded.map((e) => e[i]))),
    xStds: params.map((_, i) => {
      const s = std(encoded.map((e) => e[i]));
      return s < 1e-12 ? 1 : s;
    }),
  };
}

function normalizeVec(encoded, xMeans, xStds) {
  return encoded.map((v, i) => (v - xMeans[i]) / xStds[i]);
}

// ===== Chebyshev scalarization =====

// Compute per-row augmented-Chebyshev values with equal weights.
// Returns an array of values (higher = closer to ideal = better).
// Rows missing any objective get -Infinity.
function computeChebyshevValues(rows, objectives, rho = 0.05) {
  const stats = objectives.map((obj) => {
    const vals = rows.map((r) => +r[obj.key]).filter((v) => isFinite(v));
    if (vals.length === 0) return { min: 0, range: 1 };
    const vmin = Math.min(...vals),
      vmax = Math.max(...vals);
    return { min: vmin, range: Math.max(vmax - vmin, 1e-8) };
  });

  const lambda = 1 / objectives.length;

  return rows.map((row) => {
    const normed = objectives.map((obj, i) => {
      const v = +row[obj.key];
      if (!isFinite(v)) return null;
      const n = (v - stats[i].min) / stats[i].range;
      return obj.maximize ? n : 1 - n; // flip minimize → all "higher is better"
    });
    if (normed.some((v) => v === null)) return -Infinity;

    const diffs = normed.map((n) => lambda * (1 - n));
    const maxTerm = Math.max(...diffs);
    const sumTerm = diffs.reduce((a, b) => a + b, 0);
    return -(maxTerm + rho * sumTerm); // negate: higher y = closer to ideal
  });
}

// ===== GP model =====

// Core GP fit used by both modes.
// rows:     observation rows (all must be valid for the selected y-values).
// yValues:  one scalar per row; higher = better (Chebyshev already negated).
function buildGpModelFromValues(rows, yValues, normStats, lengthScale, params) {
  if (rows.length < 2) return null;

  const xNorm = rows.map((r) =>
    normalizeVec(encodeRow(r, params), normStats.xMeans, normStats.xStds),
  );

  const yMean = mean(yValues);
  const yStdRaw = std(yValues);
  const yStd = yStdRaw < 1e-12 ? 1 : yStdRaw;
  const yNorm = yValues.map((v) => (v - yMean) / yStd);

  const n = rows.length;
  const K = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) =>
      rbfKernel(xNorm[i], xNorm[j], lengthScale),
    ),
  );
  for (let i = 0; i < n; i++) K[i][i] += 1e-4;

  const L = choleskyDecomp(K);
  const alpha = Array.from(solveChol(L, yNorm));

  const bestIdx = yValues.reduce((bi, v, i) => (v > yValues[bi] ? i : bi), 0);

  return {
    alpha,
    xNorm,
    yMean,
    yStd,
    lengthScale,
    observations: rows,
    normStats,
    bestObs: rows[bestIdx],
    params,
  };
}

// Single-metric mode: build GP on one metric column.
function buildGpModel(rows, metricKey, normStats, lengthScale, params) {
  const metricInfo = METRICS.find((m) => m.key === metricKey);
  const valid = rows.filter(
    (r) => r[metricKey] != null && isFinite(+r[metricKey]),
  );
  if (valid.length < 2) return null;
  // Store raw values so the colorbar displays metric units.
  // bestObs picks the best direction-aware extremum.
  const yValues = valid.map((r) => +r[metricKey]);
  const model = buildGpModelFromValues(
    valid,
    metricInfo.maximize ? yValues : yValues.map((v) => -v),
    normStats,
    lengthScale,
    params,
  );
  if (!model) return null;
  // Overwrite yMean/yStd to reflect raw (non-flipped) units for the colorbar.
  if (!metricInfo.maximize) {
    model.yMean = -model.yMean;
    model.yStd = model.yStd; // magnitude unchanged
    model._flipForDisplay = true;
  }
  return model;
}

// Chebyshev mode: build GP on the scalarized Pareto objective.
function buildGpModelCheby(rows, objectives, normStats, lengthScale, params) {
  const yValues = computeChebyshevValues(rows, objectives);
  const valid = rows.filter((_, i) => isFinite(yValues[i]));
  const validY = yValues.filter((v) => isFinite(v));
  return buildGpModelFromValues(valid, validY, normStats, lengthScale, params);
}

function predictGP(model, xTestNorm) {
  const kStar = model.xNorm.map((xi) =>
    rbfKernel(xi, xTestNorm, model.lengthScale),
  );
  const rawPred = dot(kStar, model.alpha) * model.yStd + model.yMean;
  return model._flipForDisplay ? -rawPred : rawPred;
}

// Predict at a grid point (xParamIdx and yParamIdx are indices into PARAMS).
// Other params are fixed at bestObs values.
function predictAtGrid(model, xParamIdx, xv, yParamIdx, yv) {
  const tempRow = { ...model.bestObs };
  tempRow[PARAMS[xParamIdx].name] = xv;
  tempRow[PARAMS[yParamIdx].name] = yv;
  const norm = normalizeVec(
    encodeRow(tempRow, model.params),
    model.normStats.xMeans,
    model.normStats.xStds,
  );
  return predictGP(model, norm);
}

// ===== Grid helpers =====

const GRID_N = 50;

function linspace(lo, hi, n, log = false) {
  if (log) {
    const a = Math.log(lo),
      b = Math.log(hi);
    return Array.from({ length: n }, (_, i) =>
      Math.exp(a + ((b - a) * i) / (n - 1)),
    );
  }
  return Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
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
    if (a < 0.1) return v.toFixed(3);
    if (a < 1) return v.toFixed(2);
    if (a < 10) return v.toFixed(1);
    return v.toFixed(0);
  }
  if (v === 0) return "0";
  if (a < 1e-3) return v.toExponential(1);
  if (a < 1e-2) return v.toFixed(4);
  if (a < 0.1) return v.toFixed(3);
  if (a < 1) return v.toFixed(2);
  if (a < 10) return v.toFixed(1);
  return v.toFixed(0);
}

// ===== Canvas rendering =====

// Pixel margins reserved for axis labels (must match colorbar-wrapper padding in CSS)
const MARGIN = { left: 72, right: 12, top: 15, bottom: 54 };

// Draw a filled circle used for observation scatter.
function drawCircle(ctx, cx, cy, outerR, innerR, fillColor) {
  ctx.beginPath();
  ctx.arc(cx, cy, outerR, 0, 2 * Math.PI);
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(cx, cy, innerR, 0, 2 * Math.PI);
  ctx.fillStyle = fillColor;
  ctx.fill();
}

// Draw a gold circle with a white ring for Pareto front points.
function drawParetoMarker(ctx, cx, cy) {
  drawCircle(ctx, cx, cy, 8, 6, "#FFD700");
  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
  ctx.strokeStyle = "white";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function renderHeatmap(
  canvas,
  model,
  xParamIdx,
  yParamIdx,
  predictions,
  xVals,
  yVals,
  vmin,
  vmax,
  paretoFront,
) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width,
    H = canvas.height;
  const plotW = W - MARGIN.left - MARGIN.right;
  const plotH = H - MARGIN.top - MARGIN.bottom;

  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, W, H);
  if (plotW <= 0 || plotH <= 0) return;

  const xParam = PARAMS[xParamIdx],
    yParam = PARAMS[yParamIdx];
  const range = vmax - vmin;

  // Heatmap via ImageData
  const imgData = ctx.createImageData(plotW, plotH);
  const buf = imgData.data;
  for (let xi = 0; xi < GRID_N; xi++) {
    const px0 = Math.round((xi * plotW) / GRID_N);
    const px1 = Math.round(((xi + 1) * plotW) / GRID_N);
    for (let yi = 0; yi < GRID_N; yi++) {
      const t =
        range > 1e-12 ? (predictions[xi * GRID_N + yi] - vmin) / range : 0.5;
      const [r, g, b] = viridisRgb(t);
      // yi=0 is the bottom of the plot, so flip vertically
      const py0 = Math.round(((GRID_N - 1 - yi) * plotH) / GRID_N);
      const py1 = Math.round(((GRID_N - yi) * plotH) / GRID_N);
      for (let px = px0; px < px1; px++) {
        for (let py = py0; py < py1; py++) {
          const idx = (py * plotW + px) * 4;
          buf[idx] = r;
          buf[idx + 1] = g;
          buf[idx + 2] = b;
          buf[idx + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(imgData, MARGIN.left, MARGIN.top);

  // Plot border
  ctx.strokeStyle = "#555";
  ctx.lineWidth = 1;
  ctx.strokeRect(MARGIN.left + 0.5, MARGIN.top + 0.5, plotW, plotH);

  // X axis
  ctx.fillStyle = "#aaa";
  ctx.font = "11px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of linspace(xParam.min, xParam.max, 5, xParam.log)) {
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

  // Y axis
  ctx.font = "11px monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of linspace(yParam.min, yParam.max, 5, yParam.log)) {
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

  // Regular observation scatter (white dots)
  for (const obs of model.observations) {
    const xv = +obs[xParam.name],
      yv = +obs[yParam.name];
    if (!isFinite(xv) || !isFinite(yv)) continue;
    const fx = valueToFrac(xv, xParam.min, xParam.max, xParam.log);
    const fy = valueToFrac(yv, yParam.min, yParam.max, yParam.log);
    const px = MARGIN.left + Math.max(0, Math.min(1, fx)) * plotW;
    const py = MARGIN.top + (1 - Math.max(0, Math.min(1, fy))) * plotH;
    drawCircle(ctx, px, py, 5.5, 4, "white");
  }

  // Pareto front overlay (gold circles with white ring, drawn on top)
  if (paretoFront) {
    for (const entry of paretoFront) {
      const xv = +entry[xParam.name],
        yv = +entry[yParam.name];
      if (!isFinite(xv) || !isFinite(yv)) continue;
      const fx = valueToFrac(xv, xParam.min, xParam.max, xParam.log);
      const fy = valueToFrac(yv, yParam.min, yParam.max, yParam.log);
      const px = MARGIN.left + Math.max(0, Math.min(1, fx)) * plotW;
      const py = MARGIN.top + (1 - Math.max(0, Math.min(1, fy))) * plotH;
      drawParetoMarker(ctx, px, py);
    }
  }
}

function renderColorbar(canvas, vmin, vmax) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width,
    H = canvas.height;
  if (H <= 0) return;
  const imgData = ctx.createImageData(W, H);
  const buf = imgData.data;
  for (let py = 0; py < H; py++) {
    const t = 1 - py / (H - 1);
    const [r, g, b] = viridisRgb(t);
    for (let px = 0; px < W; px++) {
      const idx = (py * W + px) * 4;
      buf[idx] = r;
      buf[idx + 1] = g;
      buf[idx + 2] = b;
      buf[idx + 3] = 255;
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

  document
    .getElementById("ld-dataset")
    .addEventListener("change", populateCurvatures);
  populateCurvatures();

  document.getElementById("ld-mode").addEventListener("change", onModeChange);
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

function onModeChange() {
  const mode = document.getElementById("ld-mode").value;
  document.getElementById("metric-control").style.display =
    mode === "chebyshev" ? "none" : "";
}

function populateDatasets() {
  const datasets = [...new Set(allResults.map((r) => r.dataset_name))].sort();
  const sel = document.getElementById("ld-dataset");
  sel.innerHTML = datasets
    .map((d) => `<option value="${d}">${d}</option>`)
    .join("");
}

function populateCurvatures() {
  const dataset = document.getElementById("ld-dataset").value;
  const rows = allResults.filter((r) => r.dataset_name === dataset);
  const geometries = [
    ...new Set(rows.filter((r) => r.geometry).map((r) => r.geometry)),
  ].sort();
  document.getElementById("ld-curvature").innerHTML = geometries
    .map((g) => `<option value="${g}">${g}</option>`)
    .join("");
}

function populateParams() {
  const opts = PARAMS.map(
    (p, i) => `<option value="${i}">${p.label}</option>`,
  ).join("");
  const xSel = document.getElementById("ld-x-param");
  const ySel = document.getElementById("ld-y-param");
  xSel.innerHTML = ySel.innerHTML = opts;
  xSel.value = "0"; // learning_rate
  ySel.value = "1"; // perplexity_ratio
}

function populateMetrics() {
  document.getElementById("ld-metric").innerHTML = METRICS.map(
    (m) => `<option value="${m.key}">${m.label}</option>`,
  ).join("");
}

function syncCanvasSize() {
  const wrapper = document.querySelector(".heatmap-wrapper");
  const canvas = document.getElementById("heatmap");
  const r = wrapper.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(r.width));
  canvas.height = Math.max(1, Math.floor(r.height));
}

async function onRender() {
  const dataset = document.getElementById("ld-dataset").value;
  const geometry = document.getElementById("ld-curvature").value;
  const xParamIdx = parseInt(document.getElementById("ld-x-param").value);
  const yParamIdx = parseInt(document.getElementById("ld-y-param").value);
  const mode = document.getElementById("ld-mode").value;
  const metricKey = document.getElementById("ld-metric").value;

  if (xParamIdx === yParamIdx) {
    setStatus("X and Y axes must be different parameters.");
    return;
  }

  const obs = allResults.filter(
    (r) => r.dataset_name === dataset && r.geometry === geometry,
  );

  document.getElementById("ld-render-btn").disabled = true;
  setStatus("Computing…");

  // Load Pareto front (null if not available)
  const paretoFront = await loadParetoFront(dataset, geometry);

  // Determine valid observations for this mode
  let valid;
  if (mode === "chebyshev") {
    valid = obs.filter((r) =>
      PARETO_OBJECTIVES.every((o) => r[o.key] != null && isFinite(+r[o.key])),
    );
    if (valid.length < 2) {
      setStatus(
        `Only ${valid.length} rows have all Pareto objectives — need ≥ 2.`,
      );
      document.getElementById("ld-render-btn").disabled = false;
      return;
    }
  } else {
    valid = obs.filter((r) => r[metricKey] != null && isFinite(+r[metricKey]));
    if (valid.length < 2) {
      setStatus(
        `Only ${valid.length} valid obs for "${metricKey}" — need ≥ 2.`,
      );
      document.getElementById("ld-render-btn").disabled = false;
      return;
    }
  }

  const xParam = PARAMS[xParamIdx],
    yParam = PARAMS[yParamIdx];
  const params = activeParams(valid);
  if (!params.includes(xParam)) {
    setStatus(`X param "${xParam.label}" has no data for this group.`);
    document.getElementById("ld-render-btn").disabled = false;
    return;
  }
  if (!params.includes(yParam)) {
    setStatus(`Y param "${yParam.label}" has no data for this group.`);
    document.getElementById("ld-render-btn").disabled = false;
    return;
  }

  // Use GP state file's length_scale when available (better estimate than the default 1.0)
  let lengthScale = 1.0;
  const gpState = await loadGpState(dataset, geometry);
  if (gpState && typeof gpState.length_scale === "number") {
    lengthScale = Math.max(gpState.length_scale, 0.01);
  }

  // Always compute normalization from the current valid observations
  const normStats = computeNormStats(valid, params);
  let model;
  if (mode === "chebyshev") {
    model = buildGpModelCheby(
      valid,
      PARETO_OBJECTIVES,
      normStats,
      lengthScale,
      params,
    );
  } else {
    model = buildGpModel(valid, metricKey, normStats, lengthScale, params);
  }

  if (!model) {
    setStatus("Failed to build GP model.");
    document.getElementById("ld-render-btn").disabled = false;
    return;
  }

  // Evaluate GP on GRID_N × GRID_N grid
  const xVals = linspace(xParam.min, xParam.max, GRID_N, xParam.log);
  const yVals = linspace(yParam.min, yParam.max, GRID_N, yParam.log);
  const predictions = new Float64Array(GRID_N * GRID_N);
  for (let xi = 0; xi < GRID_N; xi++) {
    for (let yi = 0; yi < GRID_N; yi++) {
      predictions[xi * GRID_N + yi] = predictAtGrid(
        model,
        xParamIdx,
        xVals[xi],
        yParamIdx,
        yVals[yi],
      );
    }
  }

  const vmin = Math.min(...predictions),
    vmax = Math.max(...predictions);

  lastRenderState = {
    model,
    xParamIdx,
    yParamIdx,
    predictions,
    xVals,
    yVals,
    vmin,
    vmax,
    paretoFront,
    mode,
  };
  drawAll(lastRenderState);

  document.getElementById("ld-render-btn").disabled = false;

  const lsSource = gpState ? "GP state" : "default";
  const fixedParams = params.filter((p) => p !== xParam && p !== yParam);
  const frontNote = paretoFront
    ? ` · ${paretoFront.length} Pareto pts (gold)`
    : "";
  const modeNote =
    mode === "chebyshev"
      ? `Chebyshev (${PARETO_OBJECTIVES.length} obj, equal weights)`
      : `${metricKey}`;
  setStatus(
    `${valid.length} obs · ls=${lengthScale.toFixed(3)} (${lsSource}) · ${modeNote}${frontNote}\n` +
      `Fixed at best: ${fixedParams.map((p) => `${p.name}=${(+model.bestObs[p.name]).toFixed(3)}`).join(", ")}`,
  );
}

function drawAll({
  model,
  xParamIdx,
  yParamIdx,
  predictions,
  xVals,
  yVals,
  vmin,
  vmax,
  paretoFront,
}) {
  syncCanvasSize();
  const canvas = document.getElementById("heatmap");
  renderHeatmap(
    canvas,
    model,
    xParamIdx,
    yParamIdx,
    predictions,
    xVals,
    yVals,
    vmin,
    vmax,
    paretoFront,
  );

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
