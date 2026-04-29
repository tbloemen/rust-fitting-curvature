import {
  EmbeddingRunner,
  get_default_config,
  default as init,
} from "fitting-web";
import {
  parseIdxBuffers,
  subsampleMnist,
  subsampleFashionMnist,
  parsePbmcText,
  parseWordnetEdges,
} from "./dataLoaders.js";
import PARAMS from "@config/params.json";

// ---------------------------------------------------------------------------
// Pareto t-SNE constants
// ---------------------------------------------------------------------------

const PARAM_CONFIG = Object.fromEntries(PARAMS.map((p) => [p.name, p]));

const PARETO_METRICS_LIST = [
  { key: "trustworthiness", label: "Trustworthiness" },
  { key: "trustworthiness_manifold", label: "Trustworthiness (manifold)" },
  { key: "continuity", label: "Continuity" },
  { key: "continuity_manifold", label: "Continuity (manifold)" },
  { key: "normalized_stress", label: "Normalized Stress" },
  { key: "normalized_stress_manifold", label: "Normalized Stress (manifold)" },
  { key: "shepard_goodness", label: "Shepard Goodness" },
  { key: "shepard_goodness_manifold", label: "Shepard Goodness (manifold)" },
  { key: "neighborhood_hit", label: "Neighborhood Hit" },
  { key: "neighborhood_hit_manifold", label: "Neighborhood Hit (manifold)" },
];

// Tab10 palette — matches visualisation.rs tab10_color
const TAB10 = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf",
];

// Canvas margins — must match plot.rs MARGIN constants
const PLOT_MARGIN = { left: 25, right: 5, top: 5, bottom: 25 };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let dataSource = "real";
let realDataset = "mnist";
let runner = null;
let animationId = null;
// Node names for datasets that render text labels (e.g., WordNet).
// Array of strings in the same order as embedding points, or null.
let pointNames = null;
// Edges as [[src, dst], ...] in compact point-index space, or null.
let pointEdges = null;

// Pareto t-SNE state
let paretoTsneEntries = null;
let paretoActiveKeys = [];
let paretoPointBins = null;
let paretoBinEdges = null;
let paretoNBins = 8;

// Cache of raw fetched data keyed by dataset name
const rawDataCache = {};

// Pan state (in canvas pixel space)
let isPanning = false;
let panLastX = 0;
let panLastY = 0;

// DOM refs
let canvas, canvasWrapper, zoomIndicator;
let status, lossDisplay, fpsDisplay, runBtn, stopBtn, stepBtn;
let progressContainer, progressBar, metricsPanel, metricsContent;
let sidebar, sidebarToggle;
let lastFrameTime = 0;
let fpsSmoothed = 0;

initialize();

async function initialize() {
  await init();
  applyDefaultConfig();
  main();
}

function applyDefaultConfig() {
  const d = get_default_config();
  document.getElementById("curvature").value = d.curvature;
  document.getElementById("perplexity").value = d.perplexity;
  document.getElementById("iterations").value = d.n_iterations;
  document.getElementById("lr").value = d.learning_rate;
  document.getElementById("ee_factor").value = d.early_exaggeration_factor;
  document.getElementById("ee_iterations").value =
    d.early_exaggeration_iterations;
  document.getElementById("centering_weight").value = d.centering_weight;
  document.getElementById("scaling_loss").value = d.scaling_loss;
  document.getElementById("global_loss_weight").value = d.global_loss_weight;
  document.getElementById("norm_loss_weight").value = d.norm_loss_weight;
}

function main() {
  canvas = document.getElementById("canvas");
  canvasWrapper = document.getElementById("canvas-wrapper");
  zoomIndicator = document.getElementById("zoom-indicator");
  status = document.getElementById("status");
  lossDisplay = document.getElementById("loss-display");
  fpsDisplay = document.getElementById("fps-display");
  runBtn = document.getElementById("run-btn");
  stopBtn = document.getElementById("stop-btn");
  stepBtn = document.getElementById("step-btn");
  progressContainer = document.getElementById("progress-container");
  progressBar = document.getElementById("progress-bar");
  metricsPanel = document.getElementById("metrics-panel");
  metricsContent = document.getElementById("metrics-content");

  sidebar = document.getElementById("sidebar");
  sidebarToggle = document.getElementById("sidebar-toggle");

  setupCanvas();
  setupUI();
  setupZoomPan();
  setupSidebarToggle();
  setupParetoLoader();
  setupParetoTooltip();

  status.textContent = "WebAssembly loaded! Click Run to start.";
}

// Dataset notes shown below the controls
const DATASET_NOTES = {
  mnist: "Test set (10k images). Fetched from public/data/mnist/.",
  fashion_mnist:
    "Place IDX files in public/data/fashion-mnist/. Same format as MNIST.",
  wordnet_mammals:
    "Generate with: uv run python scripts/generate_wordnet_mammals.py. Place in public/data/wordnet/.",
  pbmc: "Place pbmc_pca.tsv (pre-processed PCA) in public/data/pbmc/.",
};

function updateDatasetNote() {
  const note = document.getElementById("real-dataset-note");
  note.textContent = DATASET_NOTES[realDataset] || "";
}

function setupUI() {
  document
    .getElementById("btn-real")
    .addEventListener("click", () => setDataSource("real"));
  document
    .getElementById("btn-synthetic")
    .addEventListener("click", () => setDataSource("synthetic"));
  document
    .getElementById("btn-pareto")
    .addEventListener("click", () => setDataSource("pareto"));
  document
    .getElementById("pareto-tsne-file")
    .addEventListener("change", onParetoTsneFileChange);
  document
    .getElementById("pareto-tsne-metric")
    .addEventListener("change", onParetoColorChange);
  document
    .getElementById("pareto-tsne-bins")
    .addEventListener("change", onParetoColorChange);

  const realDatasetSelect = document.getElementById("real-dataset");
  realDatasetSelect.addEventListener("change", () => {
    realDataset = realDatasetSelect.value;
    updateDatasetNote();
  });

  runBtn.addEventListener("click", runEmbedding);
  stopBtn.addEventListener("click", () => {
    if (animationId !== null) {
      stopEmbedding();
    } else {
      resetEmbedding();
    }
  });
  stepBtn.addEventListener("click", stepEmbedding);
  window.addEventListener("resize", () => {
    setupCanvas();
    if (runner !== null && animationId === null) {
      renderFrame();
    }
  });

  updateDatasetNote();
}

function setupCanvas() {
  const rect = canvasWrapper.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
}

function setupSidebarToggle() {
  sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
  });
  // Resize canvas after sidebar transition ends
  sidebar.addEventListener("transitionend", () => {
    setupCanvas();
    if (runner !== null && animationId === null) {
      renderFrame();
    }
  });
}

function setupZoomPan() {
  canvasWrapper.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      if (runner === null) return;
      const rect = canvas.getBoundingClientRect();
      const norm_x = (e.clientX - rect.left) / rect.width;
      const norm_y = (e.clientY - rect.top) / rect.height;
      const factor = e.deltaY < 0 ? 1.05 : 1 / 1.05;
      runner.zoom_at(norm_x, norm_y, factor);
      renderFrame();
    },
    { passive: false },
  );

  canvasWrapper.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    isPanning = true;
    panLastX = e.clientX;
    panLastY = e.clientY;
    canvasWrapper.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", (e) => {
    if (!isPanning || runner === null) return;
    const rect = canvas.getBoundingClientRect();
    const dx = (e.clientX - panLastX) / rect.width;
    const dy = (e.clientY - panLastY) / rect.height;
    panLastX = e.clientX;
    panLastY = e.clientY;
    runner.pan_by(dx, dy);
    renderFrame();
  });

  window.addEventListener("mouseup", () => {
    if (!isPanning) return;
    isPanning = false;
    canvasWrapper.style.cursor = "";
  });

  canvasWrapper.addEventListener("dblclick", () => {
    if (runner === null) return;
    runner.reset_view();
    renderFrame();
  });
}

function setDataSource(source) {
  dataSource = source;
  document
    .getElementById("btn-real")
    .classList.toggle("active", source === "real");
  document
    .getElementById("btn-synthetic")
    .classList.toggle("active", source === "synthetic");
  document
    .getElementById("btn-pareto")
    .classList.toggle("active", source === "pareto");
  document.getElementById("real-controls").style.display =
    source === "real" ? "block" : "none";
  document.getElementById("synthetic-controls").style.display =
    source === "synthetic" ? "block" : "none";
  document.getElementById("pareto-tsne-controls").style.display =
    source === "pareto" ? "block" : "none";
}

function getParams() {
  return {
    curvature: parseFloat(document.getElementById("curvature").value),
    perplexity: parseFloat(document.getElementById("perplexity").value),
    iterations: parseInt(document.getElementById("iterations").value),
    lr: parseFloat(document.getElementById("lr").value),
    eeFactor: parseFloat(document.getElementById("ee_factor").value),
    eeIterations: parseInt(document.getElementById("ee_iterations").value),
    centeringWeight: parseFloat(
      document.getElementById("centering_weight").value,
    ),
    scalingLoss: document.getElementById("scaling_loss").value,
    globalLossWeight: parseFloat(
      document.getElementById("global_loss_weight").value,
    ),
    normLossWeight: parseFloat(
      document.getElementById("norm_loss_weight").value,
    ),
    projection: document.getElementById("projection").value,
  };
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

async function loadMnistLike(baseUrl, nPoints) {
  if (!rawDataCache[baseUrl]) {
    const imagesUrl = `${baseUrl}/t10k-images-idx3-ubyte`;
    const labelsUrl = `${baseUrl}/t10k-labels-idx1-ubyte`;
    const [imagesResp, labelsResp] = await Promise.all([
      fetch(imagesUrl),
      fetch(labelsUrl),
    ]);
    if (!imagesResp.ok)
      throw new Error(`Could not fetch ${imagesUrl}: ${imagesResp.status}`);
    if (!labelsResp.ok)
      throw new Error(`Could not fetch ${labelsUrl}: ${labelsResp.status}`);
    const [imagesBuf, labelsBuf] = await Promise.all([
      imagesResp.arrayBuffer(),
      labelsResp.arrayBuffer(),
    ]);
    rawDataCache[baseUrl] = parseIdxBuffers(imagesBuf, labelsBuf);
  }
  if (baseUrl.endsWith("fashion-mnist")) {
    return subsampleFashionMnist(rawDataCache[baseUrl], nPoints);
  }
  return subsampleMnist(rawDataCache[baseUrl], nPoints);
}

async function loadWordnetMammals(nPoints) {
  const cacheKey = "wordnet_mammals";
  if (!rawDataCache[cacheKey]) {
    const [edgesResp, labelsResp, namesResp] = await Promise.all([
      fetch("data/wordnet/mammals_edges.tsv"),
      fetch("data/wordnet/mammals_labels.tsv"),
      fetch("data/wordnet/mammals_names.tsv"),
    ]);
    if (!edgesResp.ok)
      throw new Error(
        `Could not fetch mammals_edges.tsv: ${edgesResp.status}. ` +
          `Run: uv run python scripts/generate_wordnet_mammals.py`,
      );
    const isHtml = (resp) =>
      (resp.headers.get("content-type") || "").includes("text/html");
    rawDataCache[cacheKey] = {
      edgesText: await edgesResp.text(),
      labelsText:
        labelsResp.ok && !isHtml(labelsResp) ? await labelsResp.text() : null,
      namesText:
        namesResp.ok && !isHtml(namesResp) ? await namesResp.text() : null,
    };
  }
  const { edgesText, labelsText, namesText } = rawDataCache[cacheKey];
  return parseWordnetEdges(edgesText, labelsText, namesText, nPoints);
}

async function loadPbmc(nPoints) {
  const cacheKey = "pbmc";
  if (!rawDataCache[cacheKey]) {
    const resp = await fetch("data/pbmc/pbmc_pca.tsv");
    if (!resp.ok)
      throw new Error(
        `Could not fetch pbmc_pca.tsv: ${resp.status}. ` +
          `Place pre-processed PCA TSV in www/public/data/pbmc/pbmc_pca.tsv`,
      );
    rawDataCache[cacheKey] = await resp.text();
  }
  return parsePbmcText(rawDataCache[cacheKey], nPoints);
}

// ---------------------------------------------------------------------------
// Text label overlay (WordNet names)
// ---------------------------------------------------------------------------

/**
 * Draw tree edges and node name labels on top of the plotters render.
 *
 * Edges are always drawn (thin, semi-transparent).
 * Labels are drawn with greedy bounding-box occupancy culling: a label is
 * skipped if it would overlap any already-committed label, preventing
 * unreadable pileups when points are dense.
 *
 * Chart margins in plot.rs: left=25, right=5, top=5, bottom=25 (px).
 */
function drawNameOverlay() {
  if (!runner) return;
  if (!pointNames && !pointEdges) return;

  const coords = runner.get_projected_coords(); // [x0,y0,x1,y1,...]
  const vp = runner.get_viewport(); // [cx, cy, half, auto_half]
  const [vpCx, vpCy, vpHalf] = vp;

  const w = canvas.width;
  const h = canvas.height;
  const aspect = w / h;
  const halfX = vpHalf * aspect;

  const MARGIN_LEFT = 25;
  const MARGIN_RIGHT = 5;
  const MARGIN_TOP = 5;
  const MARGIN_BOTTOM = 25;
  const plotW = w - MARGIN_LEFT - MARGIN_RIGHT;
  const plotH = h - MARGIN_TOP - MARGIN_BOTTOM;

  const xMin = vpCx - halfX;
  const xMax = vpCx + halfX;
  const yMin = vpCy - vpHalf;
  const yMax = vpCy + vpHalf;

  /** Convert a plot-space point to canvas device pixels. */
  function toCanvas(px, py) {
    return [
      MARGIN_LEFT + ((px - xMin) / (xMax - xMin)) * plotW,
      h - MARGIN_BOTTOM - ((py - yMin) / (yMax - yMin)) * plotH,
    ];
  }

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const n = coords.length / 2;

  // --- Edges ---
  if (pointEdges) {
    ctx.save();
    ctx.strokeStyle = "rgba(100, 100, 100, 0.35)";
    ctx.lineWidth = dpr;
    ctx.beginPath();
    for (const [a, b] of pointEdges) {
      const ax = coords[a * 2],
        ay = coords[a * 2 + 1];
      const bx = coords[b * 2],
        by = coords[b * 2 + 1];
      if (!isFinite(ax) || !isFinite(ay) || !isFinite(bx) || !isFinite(by))
        continue;
      // Skip edges entirely outside the viewport
      if (ax < xMin && bx < xMin) continue;
      if (ax > xMax && bx > xMax) continue;
      if (ay < yMin && by < yMin) continue;
      if (ay > yMax && by > yMax) continue;
      const [cax, cay] = toCanvas(ax, ay);
      const [cbx, cby] = toCanvas(bx, by);
      ctx.moveTo(cax, cay);
      ctx.lineTo(cbx, cby);
    }
    ctx.stroke();
    ctx.restore();
  }

  // --- Labels with occupancy culling ---
  if (pointNames) {
    const fontSize = Math.max(9, Math.round(10 * dpr));
    ctx.save();
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillStyle = "rgba(20, 20, 20, 0.85)";
    ctx.textBaseline = "bottom";

    const offsetX = 4 * dpr;
    const offsetY = 1 * dpr;
    const padding = 2 * dpr;

    // Committed label bounding boxes for overlap detection
    const occupied = [];

    function overlaps(r) {
      for (const o of occupied) {
        if (
          r.x < o.x + o.w &&
          r.x + r.w > o.x &&
          r.y < o.y + o.h &&
          r.y + r.h > o.y
        )
          return true;
      }
      return false;
    }

    for (let i = 0; i < n; i++) {
      const name = pointNames[i];
      if (!name) continue;
      const px = coords[i * 2],
        py = coords[i * 2 + 1];
      if (!isFinite(px) || !isFinite(py)) continue;
      if (px < xMin || px > xMax || py < yMin || py > yMax) continue;

      const [cx, cy] = toCanvas(px, py);
      const tw = ctx.measureText(name).width;
      const rect = {
        x: cx + offsetX - padding,
        y: cy - fontSize - offsetY - padding,
        w: tw + padding * 2,
        h: fontSize + padding * 2,
      };

      if (!overlaps(rect)) {
        ctx.fillText(name, cx + offsetX, cy - offsetY);
        occupied.push(rect);
      }
    }

    ctx.restore();
  }
}

/** Render the current embedding state and apply any text/color overlay. */
function renderFrame() {
  runner.render();
  drawNameOverlay();
  if (dataSource === "pareto") drawParetoColorOverlay();
}

// ---------------------------------------------------------------------------
// Runner creation
// ---------------------------------------------------------------------------

function updateTitle(curvature, projection) {
  if (dataSource === "pareto") return;
  const el = document.getElementById("plot-title");
  if (curvature < 0) {
    el.textContent = `Hyperbolic (k=${curvature}) \u2014 Poincar\u00e9 disk`;
  } else if (curvature > 0) {
    const projNames = {
      stereographic: "Stereographic",
      azimuthal_equidistant: "Azimuthal equidistant",
      orthographic: "Orthographic",
    };
    el.textContent = `Spherical (k=${curvature}) \u2014 ${projNames[projection] || projection}`;
  } else {
    el.textContent = "Euclidean";
  }
}

async function createRunner() {
  if (runner !== null) {
    runner.free();
    runner = null;
  }
  pointNames = null;
  pointEdges = null;

  const p = getParams();
  updateTitle(p.curvature, p.projection);

  if (dataSource === "real") {
    const nPoints = parseInt(document.getElementById("real_n_points").value);
    status.textContent = `Loading ${realDataset}…`;

    const commonArgs = [
      "canvas",
      p.curvature,
      p.iterations,
      p.perplexity,
      p.lr,
      p.eeFactor,
      p.eeIterations,
      p.centeringWeight,
      p.scalingLoss,
      p.globalLossWeight,
      p.normLossWeight,
      p.projection,
    ];

    if (realDataset === "mnist") {
      const d = await loadMnistLike("data/mnist", nPoints);
      runner = EmbeddingRunner.from_data_with_labels(
        ...commonArgs.slice(0, 1),
        d.data,
        d.labels,
        d.nPoints,
        d.nFeatures,
        ...commonArgs.slice(1),
      );
      if (d.labelNames) runner.set_label_names(d.labelNames.join("\t"));
    } else if (realDataset === "fashion_mnist") {
      const d = await loadMnistLike("data/fashion-mnist", nPoints);
      runner = EmbeddingRunner.from_data_with_labels(
        ...commonArgs.slice(0, 1),
        d.data,
        d.labels,
        d.nPoints,
        d.nFeatures,
        ...commonArgs.slice(1),
      );
      if (d.labelNames) runner.set_label_names(d.labelNames.join("\t"));
    } else if (realDataset === "wordnet_mammals") {
      const d = await loadWordnetMammals(nPoints);
      runner = EmbeddingRunner.from_distances(
        ...commonArgs.slice(0, 1),
        d.distances,
        d.labels,
        d.nPoints,
        ...commonArgs.slice(1),
      );
      pointNames = d.names && d.names.some((n) => n) ? d.names : null;
      pointEdges = d.edges && d.edges.length > 0 ? d.edges : null;
    } else if (realDataset === "pbmc") {
      const d = await loadPbmc(nPoints);
      runner = EmbeddingRunner.from_data_with_labels(
        ...commonArgs.slice(0, 1),
        d.data,
        d.labels,
        d.nPoints,
        d.nFeatures,
        ...commonArgs.slice(1),
      );
      if (d.labelNames) runner.set_label_names(d.labelNames.join("\t"));
    }
  } else if (dataSource === "synthetic") {
    const dataset = document.getElementById("dataset").value;
    const nPoints = parseInt(document.getElementById("synth_n_points").value);
    runner = EmbeddingRunner.from_synthetic(
      "canvas",
      dataset,
      nPoints,
      p.curvature,
      p.iterations,
      p.perplexity,
      p.lr,
      p.eeFactor,
      p.eeIterations,
      p.centeringWeight,
      p.scalingLoss,
      p.globalLossWeight,
      p.normLossWeight,
      p.projection,
    );
  } else if (dataSource === "pareto") {
    if (!paretoTsneEntries) {
      status.textContent = "No Pareto JSON loaded. Pick a file above.";
      return;
    }
    const { data, nPoints, nFeatures } = buildParetoFeatureMatrix();
    const perplexity = Math.min(p.perplexity, nPoints - 1);
    // All-zero labels — coloring is done via JS overlay in drawParetoColorOverlay()
    const labels = new Uint32Array(nPoints);
    runner = EmbeddingRunner.from_data_with_labels(
      "canvas",
      data,
      labels,
      nPoints,
      nFeatures,
      0.0, // curvature = 0 → Euclidean
      p.iterations,
      perplexity,
      p.lr,
      p.eeFactor,
      p.eeIterations,
      0.0, // centering_weight — not meaningful for parameter space
      "none",
      0.0,
      0.0,
      "stereographic",
    );
    document.getElementById("plot-title").textContent =
      "Pareto Parameter Space — Euclidean t-SNE";
  }
}

function updateDisplay() {
  const iter = runner.iteration();
  const total = runner.total_iterations();
  const loss = runner.loss();
  lossDisplay.textContent = `iter=${iter}/${total}  loss=${loss.toFixed(4)}`;
  const pct = (iter / total) * 100;
  progressBar.style.width = pct + "%";
}

function stopEmbedding() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  stopBtn.textContent = "Reset";
  stopBtn.disabled = false;
  runBtn.disabled = false;
  stepBtn.disabled = false;
  const iter = runner ? runner.iteration() : 0;
  status.textContent = `Stopped at iteration ${iter}`;
}

function resetEmbedding() {
  if (runner !== null) {
    runner.free();
    runner = null;
  }
  stopBtn.textContent = "Stop";
  stopBtn.disabled = true;
  runBtn.disabled = false;
  stepBtn.disabled = false;
  lossDisplay.textContent = "";
  progressContainer.style.display = "none";
  metricsPanel.style.display = "none";
  status.textContent = "Reset. Click Run or Step to start.";
}

// Metric groups: each dual-variant metric shows manifold + 2D columns.
// Single-variant metrics (labels-only, 2D-only) show one value column.
const METRIC_GROUPS = [
  {
    title: "Local Structure",
    dual: true,
    metrics: [
      { key: "trustworthiness", label: "Trustworthiness", dir: "↑" },
      { key: "continuity", label: "Continuity", dir: "↑" },
      { key: "knn_overlap", label: "KNN Overlap", dir: "↑" },
      { key: "neighborhood_hit", label: "Neighborhood Hit", dir: "↑" },
    ],
  },
  {
    title: "Distance Preservation",
    dual: true,
    metrics: [
      { key: "normalized_stress", label: "Norm. Stress", dir: "↓" },
      { key: "shepard_goodness", label: "Shepard Goodness", dir: "↑" },
    ],
  },
  {
    title: "Class Separation (2D)",
    dual: false,
    metrics: [
      { key: "class_density_measure", label: "Class Density", dir: "↑" },
      { key: "cluster_density_measure", label: "Cluster Density", dir: "↑" },
      { key: "davies_bouldin_ratio", label: "DB Ratio", dir: "↑" },
    ],
  },
];

// Shared inline styles for metric rows. Inline styles are used because
// .metrics-panel lives inside .canvas-wrapper which sets line-height:0;
// the panel resets that in CSS, but these row styles are straightforward
// enough to keep here alongside the HTML generation logic.
const MS = {
  row: "display:flex;align-items:center;padding:2px 0;gap:6px;",
  name: "flex-grow:1;flex-shrink:1;flex-basis:auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#ccc;font-size:0.75rem;",
  val: "flex-grow:0;flex-shrink:0;flex-basis:58px;text-align:right;color:#5dade2;font-family:monospace;font-size:0.75rem;white-space:nowrap;",
  hdr: "flex-grow:0;flex-shrink:0;flex-basis:58px;text-align:right;color:#aaa;font-size:0.6rem;white-space:nowrap;",
  title:
    "font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;color:#888;padding:8px 0 3px;border-bottom:1px solid #2a2a4a;margin-bottom:2px;",
};

function showMetrics() {
  try {
    const m = runner.compute_metrics();
    let html = "";

    for (const group of METRIC_GROUPS) {
      const rows = group.metrics.filter((mt) =>
        group.dual
          ? m[mt.key + "_manifold"] !== undefined
          : m[mt.key] !== undefined,
      );
      if (rows.length === 0) continue;

      html += `<div style="${MS.title}">${group.title}</div>`;

      if (group.dual) {
        html += `<div style="${MS.row}"><span style="${MS.name}"></span><span style="${MS.hdr}">Manifold</span><span style="${MS.hdr}">2D</span></div>`;
        for (const mt of rows) {
          const vm = m[mt.key + "_manifold"].toFixed(4);
          const v2 = m[mt.key + "_2d"].toFixed(4);
          html += `<div style="${MS.row}" title="${mt.dir} better"><span style="${MS.name}">${mt.label} ${mt.dir}</span><span style="${MS.val}">${vm}</span><span style="${MS.val}">${v2}</span></div>`;
        }
      } else {
        for (const mt of rows) {
          const v = m[mt.key].toFixed(4);
          html += `<div style="${MS.row}" title="${mt.dir} better"><span style="${MS.name}">${mt.label} ${mt.dir}</span><span style="${MS.val}">${v}</span></div>`;
        }
      }
    }

    metricsContent.innerHTML = html;
    metricsPanel.style.display = "block";
  } catch (e) {
    console.error("Metrics computation failed:", e);
  }
}

async function stepEmbedding() {
  // Stop any running animation first
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  try {
    // Create runner if not yet initialized — show initial state (step 0)
    if (runner === null || runner.is_done()) {
      await createRunner();
      renderFrame();
      updateDisplay();
      progressContainer.style.display = "block";
      metricsPanel.style.display = "none";
      stopBtn.textContent = "Reset";
      stopBtn.disabled = false;
      runBtn.disabled = false;
      stepBtn.disabled = false;
      status.textContent = "Paused (step mode) — initial state";
      return;
    }

    const hasMore = runner.step(1);
    renderFrame();
    updateDisplay();

    if (hasMore) {
      stopBtn.textContent = "Reset";
      stopBtn.disabled = false;
      runBtn.disabled = false;
      stepBtn.disabled = false;
      status.textContent = "Paused (step mode)";
    } else {
      status.textContent = `Done (${runner.iteration()} iterations)`;
      stopBtn.textContent = "Reset";
      stopBtn.disabled = false;
      stepBtn.disabled = true;
      showMetrics();
    }
  } catch (e) {
    status.textContent = "Error: " + e;
    console.error(e);
  }
}

// ---------------------------------------------------------------------------
// Pareto front loader
// ---------------------------------------------------------------------------

let paretoEntries = null;

function setupParetoLoader() {
  const fileInput = document.getElementById("pareto-file");
  const geometryRow = document.getElementById("pareto-geometry-row");
  const entryRow = document.getElementById("pareto-entry-row");
  const entrySelect = document.getElementById("pareto-entry");
  const metricsPreview = document.getElementById("pareto-metrics-preview");
  const applyBtn = document.getElementById("pareto-apply-btn");

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    try {
      const text = await file.text();
      paretoEntries = JSON.parse(text);
      if (!Array.isArray(paretoEntries) || paretoEntries.length === 0) {
        status.textContent = "Pareto JSON must be a non-empty array.";
        return;
      }

      // Try to infer geometry from filename
      const name = file.name.toLowerCase();
      const geoSelect = document.getElementById("pareto-geometry");
      if (name.includes("hyperbolic")) geoSelect.value = "-1";
      else if (name.includes("spherical")) geoSelect.value = "1";
      else geoSelect.value = "0";

      // Populate entry selector
      entrySelect.innerHTML = "";
      paretoEntries.forEach((entry, i) => {
        const opt = document.createElement("option");
        opt.value = i;
        const trust =
          entry.metrics?.trustworthiness != null
            ? `T=${entry.metrics.trustworthiness.toFixed(3)}`
            : "";
        const stress =
          entry.metrics?.normalized_stress != null
            ? `S=${entry.metrics.normalized_stress.toFixed(3)}`
            : "";
        opt.textContent = `#${i + 1}  k=${entry.curvature_magnitude?.toFixed(3) ?? "?"}  lr=${entry.learning_rate?.toFixed(3) ?? "?"}  ${trust}  ${stress}`;
        entrySelect.appendChild(opt);
      });

      geometryRow.style.display = "block";
      entryRow.style.display = "block";
      applyBtn.style.display = "block";
      updateParetoPreview();
    } catch (e) {
      status.textContent = "Failed to parse pareto JSON: " + e.message;
    }
  });

  entrySelect.addEventListener("change", updateParetoPreview);

  applyBtn.addEventListener("click", applyParetoEntry);
}

function updateParetoPreview() {
  const entrySelect = document.getElementById("pareto-entry");
  const metricsPreview = document.getElementById("pareto-metrics-preview");
  if (!paretoEntries) return;
  const entry = paretoEntries[parseInt(entrySelect.value)];
  if (!entry) return;

  const m = entry.metrics || {};
  const lines = [];
  if (m.trustworthiness != null)
    lines.push(`Trust: ${m.trustworthiness.toFixed(4)}`);
  if (m.continuity != null)
    lines.push(`Continuity: ${m.continuity.toFixed(4)}`);
  if (m.normalized_stress != null)
    lines.push(`Stress: ${m.normalized_stress.toFixed(4)}`);
  if (m.shepard_goodness != null)
    lines.push(`Shepard: ${m.shepard_goodness.toFixed(4)}`);
  if (m.neighborhood_hit != null)
    lines.push(`NH: ${m.neighborhood_hit.toFixed(4)}`);
  metricsPreview.innerHTML = lines.join(" &nbsp;|&nbsp; ");
  metricsPreview.style.display = lines.length > 0 ? "block" : "none";
}

function applyParetoEntry() {
  const entrySelect = document.getElementById("pareto-entry");
  if (!paretoEntries) return;
  const entry = paretoEntries[parseInt(entrySelect.value)];
  if (!entry) return;

  const geoSign = parseFloat(document.getElementById("pareto-geometry").value);
  const curvature = geoSign * (entry.curvature_magnitude ?? 0);
  const nSamples =
    entry.n_samples ??
    (parseInt(document.getElementById("real_n_points").value) || 1000);
  const perplexity = (entry.perplexity_ratio ?? 0.01) * nSamples;

  document.getElementById("curvature").value = curvature;
  document.getElementById("perplexity").value = Math.round(perplexity);
  document.getElementById("lr").value = entry.learning_rate ?? 10;
  document.getElementById("centering_weight").value =
    entry.centering_weight ?? 0;
  document.getElementById("global_loss_weight").value =
    entry.global_loss_weight ?? 0;
  document.getElementById("norm_loss_weight").value =
    entry.norm_loss_weight ?? 0;

  status.textContent = `Applied pareto entry #${parseInt(entrySelect.value) + 1}. Click Run to visualise.`;
}

async function runEmbedding() {
  // Cancel any running animation
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  // Create a fresh runner only if none exists or it's done
  const resuming = runner !== null && !runner.is_done();
  if (!resuming) {
    try {
      await createRunner();
    } catch (e) {
      status.textContent = "Error: " + e;
      console.error(e);
      return;
    }
  }

  runBtn.disabled = true;
  stopBtn.textContent = "Stop";
  stopBtn.disabled = false;
  stepBtn.disabled = true;
  status.textContent = "Running...";
  if (!resuming) {
    lossDisplay.textContent = "";
    metricsPanel.style.display = "none";
  }
  progressContainer.style.display = "block";
  progressBar.style.width = "0%";

  const t0 = performance.now();

  lastFrameTime = performance.now();
  fpsSmoothed = 0;

  function animate() {
    const now = performance.now();
    const dt = now - lastFrameTime;
    lastFrameTime = now;
    if (dt > 0) {
      const instantFps = 1000 / dt;
      fpsSmoothed =
        fpsSmoothed === 0 ? instantFps : fpsSmoothed * 0.9 + instantFps * 0.1;
      fpsDisplay.textContent = `${Math.round(fpsSmoothed)} FPS`;
    }

    const hasMore = runner.step(1);
    renderFrame();
    updateDisplay();

    if (hasMore) {
      animationId = requestAnimationFrame(animate);
    } else {
      animationId = null;
      const elapsed = Math.ceil(performance.now() - t0);
      status.textContent = `Done in ${elapsed}ms — computing metrics...`;
      progressBar.style.width = "100%";

      // Compute metrics asynchronously (via setTimeout to let UI update)
      setTimeout(() => {
        if (dataSource !== "pareto") showMetrics();
        status.textContent = `Done in ${elapsed}ms (${runner.iteration()} iterations)`;
      }, 10);

      fpsDisplay.textContent = "";
      runBtn.disabled = false;
      stopBtn.textContent = "Reset";
      stopBtn.disabled = false;
      stepBtn.disabled = true;
    }
  }

  animationId = requestAnimationFrame(animate);
}

// ---------------------------------------------------------------------------
// Pareto t-SNE — data loading and coloring
// ---------------------------------------------------------------------------

async function onParetoTsneFileChange() {
  const file = document.getElementById("pareto-tsne-file").files[0];
  if (!file) return;
  const statusEl = document.getElementById("pareto-tsne-status");
  try {
    const data = JSON.parse(await file.text());
    if (!Array.isArray(data) || data.length === 0)
      throw new Error("JSON must be a non-empty array");

    paretoTsneEntries = data;

    // Detect numeric parameter keys that actually vary (exclude 'metrics')
    const candidates = Object.keys(data[0]).filter(
      (k) => k !== "metrics" && typeof data[0][k] === "number",
    );
    paretoActiveKeys = candidates.filter((key) => {
      const vals = data.map((e) => +e[key]);
      const m = vals.reduce((a, b) => a + b, 0) / vals.length;
      return vals.some((v) => Math.abs(v - m) > 1e-12);
    });

    // Populate metric selector
    const sel = document.getElementById("pareto-tsne-metric");
    const available = PARETO_METRICS_LIST.filter(
      (m) => data[0].metrics?.[m.key] != null,
    );
    sel.innerHTML = available
      .map((m) => `<option value="${m.key}">${m.label}</option>`)
      .join("");

    // Sensible default perplexity
    document.getElementById("perplexity").value = Math.max(
      2,
      Math.min(15, Math.floor(data.length / 5)),
    );

    document.getElementById("pareto-tsne-metric-row").style.display = "block";
    document.getElementById("pareto-tsne-bins-row").style.display = "block";
    statusEl.textContent = `${data.length} entries · ${paretoActiveKeys.length} parameter features`;

    computeParetoBins();
    updateParetoLegend();
  } catch (e) {
    statusEl.textContent = "Error: " + e.message;
  }
}

function onParetoColorChange() {
  computeParetoBins();
  updateParetoLegend();
  if (runner && animationId === null && dataSource === "pareto") renderFrame();
}

function computeParetoBins() {
  if (!paretoTsneEntries) return;
  const metricKey = document.getElementById("pareto-tsne-metric").value;
  paretoNBins = Math.max(
    2,
    parseInt(document.getElementById("pareto-tsne-bins").value) || 8,
  );
  const n = paretoTsneEntries.length;

  const values = paretoTsneEntries.map((e) => {
    const v = e.metrics?.[metricKey];
    return v != null && isFinite(+v) ? +v : NaN;
  });
  const sorted = values.filter((v) => isFinite(v)).sort((a, b) => a - b);

  paretoBinEdges = [];
  for (let i = 0; i <= paretoNBins; i++) {
    const q = (i / paretoNBins) * (sorted.length - 1);
    const lo = Math.floor(q),
      hi = Math.ceil(q),
      frac = q - lo;
    paretoBinEdges.push(
      sorted[lo] !== undefined
        ? sorted[lo] + frac * ((sorted[hi] ?? sorted[lo]) - sorted[lo])
        : 0,
    );
  }

  paretoPointBins = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (!isFinite(v)) {
      paretoPointBins[i] = -1;
      continue;
    }
    let bin = 0;
    for (let b = 1; b < paretoNBins; b++) {
      if (v >= paretoBinEdges[b]) bin = b;
    }
    paretoPointBins[i] = bin;
  }
}

function updateParetoLegend() {
  if (!paretoBinEdges || paretoBinEdges.length === 0) return;
  const legend = document.getElementById("pareto-tsne-legend");
  legend.innerHTML = "";
  for (let i = 0; i < paretoNBins; i++) {
    const lo = paretoBinEdges[i]?.toFixed(3) ?? "";
    const hi = paretoBinEdges[i + 1]?.toFixed(3) ?? "";
    const color = TAB10[i % TAB10.length];
    const entry = document.createElement("div");
    entry.style.cssText =
      "display:flex;align-items:center;gap:6px;font-size:0.68rem;color:#ccc;margin-bottom:2px;";
    entry.innerHTML =
      `<span style="width:11px;height:11px;border-radius:2px;flex-shrink:0;background:${color};border:1px solid rgba(255,255,255,0.2);display:inline-block;"></span>` +
      `<span>${lo}–${hi}</span>`;
    legend.appendChild(entry);
  }
  document.getElementById("pareto-tsne-legend-section").style.display = "block";
}

function buildParetoFeatureMatrix() {
  const n = paretoTsneEntries.length;
  const d = paretoActiveKeys.length;

  const encoded = paretoTsneEntries.map((e) =>
    paretoActiveKeys.map((k) => {
      const cfg = PARAM_CONFIG[k];
      const v = +e[k];
      return cfg?.log ? Math.log(Math.max(v, 1e-15)) : v;
    }),
  );

  const means = paretoActiveKeys.map(
    (_, i) => encoded.reduce((s, r) => s + r[i], 0) / n,
  );
  const stds = paretoActiveKeys.map((_, i) => {
    const variance =
      encoded.reduce((s, r) => s + (r[i] - means[i]) ** 2, 0) / n;
    const s = Math.sqrt(variance);
    return s < 1e-12 ? 1 : s;
  });

  const data = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      data[i * d + j] = (encoded[i][j] - means[j]) / stds[j];
    }
  }
  return { data, nPoints: n, nFeatures: d };
}

// ---------------------------------------------------------------------------
// Pareto t-SNE — canvas overlay and tooltip
// ---------------------------------------------------------------------------

/** Compute viewport geometry shared by overlay and tooltip. */
function getParetoViewport() {
  const w = canvas.width,
    h = canvas.height;
  const [vpCx, vpCy, vpHalf] = runner.get_viewport();
  const halfX = vpHalf * (w / h);
  const plotW = w - PLOT_MARGIN.left - PLOT_MARGIN.right;
  const plotH = h - PLOT_MARGIN.top - PLOT_MARGIN.bottom;
  return {
    w,
    h,
    plotW,
    plotH,
    xMin: vpCx - halfX,
    xMax: vpCx + halfX,
    yMin: vpCy - vpHalf,
    yMax: vpCy + vpHalf,
  };
}

function paretoToCanvas(vp, px, py) {
  return [
    PLOT_MARGIN.left + ((px - vp.xMin) / (vp.xMax - vp.xMin)) * vp.plotW,
    vp.h -
      PLOT_MARGIN.bottom -
      ((py - vp.yMin) / (vp.yMax - vp.yMin)) * vp.plotH,
  ];
}

function paretoFromCanvas(vp, cx, cy) {
  return [
    vp.xMin + ((cx - PLOT_MARGIN.left) / vp.plotW) * (vp.xMax - vp.xMin),
    vp.yMin +
      ((vp.h - PLOT_MARGIN.bottom - cy) / vp.plotH) * (vp.yMax - vp.yMin),
  ];
}

/** Draw metric-colored circles over the WASM single-color dots. */
function drawParetoColorOverlay() {
  if (!runner || !paretoPointBins) return;
  const coords = runner.get_projected_coords();
  const n = coords.length / 2;
  const ctx = canvas.getContext("2d");
  const vp = getParetoViewport();
  const dpr = window.devicePixelRatio || 1;
  const r = 5 * dpr; // slightly larger than WASM dots (radius 3) to cover them

  for (let i = 0; i < n; i++) {
    const px = coords[i * 2],
      py = coords[i * 2 + 1];
    if (!isFinite(px) || !isFinite(py)) continue;
    if (px < vp.xMin || px > vp.xMax || py < vp.yMin || py > vp.yMax) continue;
    const [cx, cy] = paretoToCanvas(vp, px, py);
    const bin = paretoPointBins[i];
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.fillStyle = bin >= 0 ? TAB10[bin % TAB10.length] : "#888";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.strokeStyle = "rgba(0,0,0,0.4)";
    ctx.lineWidth = 0.8 * dpr;
    ctx.stroke();
  }
}

function setupParetoTooltip() {
  const tooltip = document.getElementById("pareto-tsne-tooltip");

  canvasWrapper.addEventListener("mousemove", (e) => {
    if (dataSource !== "pareto" || !runner || isPanning || !paretoTsneEntries) {
      tooltip.style.display = "none";
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const mouseDevX = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const mouseDevY = ((e.clientY - rect.top) / rect.height) * canvas.height;

    const vp = getParetoViewport();
    const [plotX, plotY] = paretoFromCanvas(vp, mouseDevX, mouseDevY);

    const coords = runner.get_projected_coords();
    const n = coords.length / 2;
    let nearIdx = -1,
      nearDist = Infinity;
    for (let i = 0; i < n; i++) {
      const dx = coords[i * 2] - plotX;
      const dy = coords[i * 2 + 1] - plotY;
      const d = dx * dx + dy * dy;
      if (d < nearDist) {
        nearDist = d;
        nearIdx = i;
      }
    }

    // Threshold: 5% of plot height in plot-space
    const threshold = ((vp.yMax - vp.yMin) * 0.05) ** 2;
    if (nearIdx < 0 || nearDist > threshold) {
      tooltip.style.display = "none";
      return;
    }

    const entry = paretoTsneEntries[nearIdx];
    const metricKey = document.getElementById("pareto-tsne-metric").value;
    const metricLabel =
      PARETO_METRICS_LIST.find((m) => m.key === metricKey)?.label ?? metricKey;

    // Build tooltip: all numeric non-metrics fields, then selected metric
    const paramLines = paretoActiveKeys.map((k) => {
      const cfg = PARAM_CONFIG[k];
      return `${cfg?.label ?? k}: ${(+entry[k]).toPrecision(4)}`;
    });
    // Also show any constant fields (informational)
    const allNumericKeys = Object.keys(entry).filter(
      (k) => k !== "metrics" && typeof entry[k] === "number",
    );
    const constantLines = allNumericKeys
      .filter((k) => !paretoActiveKeys.includes(k))
      .map((k) => {
        const cfg = PARAM_CONFIG[k];
        return `${cfg?.label ?? k}: ${(+entry[k]).toPrecision(4)}`;
      });
    const metricLine =
      entry.metrics?.[metricKey] != null
        ? `<b style="color:#5dade2">${metricLabel}: ${(+entry.metrics[metricKey]).toFixed(4)}</b>`
        : "";

    const allLines = [...paramLines, ...constantLines];
    if (metricLine) allLines.push(metricLine);
    tooltip.innerHTML = allLines.join("<br>");
    tooltip.style.display = "block";

    const wRect = canvasWrapper.getBoundingClientRect();
    let tx = e.clientX - wRect.left + 14;
    let ty = e.clientY - wRect.top - 10;
    if (tx + 260 > wRect.width) tx -= 260 + 28;
    tooltip.style.left = `${tx}px`;
    tooltip.style.top = `${ty}px`;
  });

  canvasWrapper.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
  });
}
