import {
  EmbeddingRunner,
  get_default_config,
  default as init,
} from "fitting-web";
import {
  parseIdxBuffers,
  subsampleIdx,
  parseCifar10Buffer,
  parsePbmcText,
  parseWordnetEdges,
} from "./dataLoaders.js";

// State
let dataSource = "real";
let realDataset = "mnist";
let runner = null;
let animationId = null;
// Node names for datasets that render text labels (e.g., WordNet).
// Array of strings in the same order as embedding points, or null.
let pointNames = null;
// Edges as [[src, dst], ...] in compact point-index space, or null.
let pointEdges = null;

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
  document.getElementById("real-controls").style.display =
    source === "real" ? "block" : "none";
  document.getElementById("synthetic-controls").style.display =
    source === "synthetic" ? "block" : "none";
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
  return subsampleIdx(rawDataCache[baseUrl], nPoints);
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
    const isHtml = (resp) => (resp.headers.get("content-type") || "").includes("text/html");
    rawDataCache[cacheKey] = {
      edgesText: await edgesResp.text(),
      labelsText: labelsResp.ok && !isHtml(labelsResp) ? await labelsResp.text() : null,
      namesText: namesResp.ok && !isHtml(namesResp) ? await namesResp.text() : null,
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
  const vp = runner.get_viewport();             // [cx, cy, half, auto_half]
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
      const ax = coords[a * 2], ay = coords[a * 2 + 1];
      const bx = coords[b * 2], by = coords[b * 2 + 1];
      if (!isFinite(ax) || !isFinite(ay) || !isFinite(bx) || !isFinite(by)) continue;
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
        if (r.x < o.x + o.w && r.x + r.w > o.x && r.y < o.y + o.h && r.y + r.h > o.y)
          return true;
      }
      return false;
    }

    for (let i = 0; i < n; i++) {
      const name = pointNames[i];
      if (!name) continue;
      const px = coords[i * 2], py = coords[i * 2 + 1];
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

/** Render the current embedding state and apply any text overlay. */
function renderFrame() {
  runner.render();
  drawNameOverlay();
}

// ---------------------------------------------------------------------------
// Runner creation
// ---------------------------------------------------------------------------

function updateTitle(curvature, projection) {
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
        d.data, d.labels, d.nPoints, d.nFeatures,
        ...commonArgs.slice(1),
      );
    } else if (realDataset === "fashion_mnist") {
      const d = await loadMnistLike("data/fashion-mnist", nPoints);
      runner = EmbeddingRunner.from_data_with_labels(
        ...commonArgs.slice(0, 1),
        d.data, d.labels, d.nPoints, d.nFeatures,
        ...commonArgs.slice(1),
      );
    } else if (realDataset === "wordnet_mammals") {
      const d = await loadWordnetMammals(nPoints);
      runner = EmbeddingRunner.from_distances(
        ...commonArgs.slice(0, 1),
        d.distances, d.labels, d.nPoints,
        ...commonArgs.slice(1),
      );
      pointNames = d.names && d.names.some((n) => n) ? d.names : null;
      pointEdges = d.edges && d.edges.length > 0 ? d.edges : null;
    } else if (realDataset === "pbmc") {
      const d = await loadPbmc(nPoints);
      runner = EmbeddingRunner.from_data_with_labels(
        ...commonArgs.slice(0, 1),
        d.data, d.labels, d.nPoints, d.nFeatures,
        ...commonArgs.slice(1),
      );
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

const METRIC_LABELS = {
  trustworthiness: ["Trustworthiness", "higher is better"],
  continuity: ["Continuity", "higher is better"],
  knn_overlap: ["KNN Overlap", "higher is better"],
  class_density_measure: ["Class Density (CDM)", "higher is better"],
  cluster_density_measure: ["Cluster Density (ClDM)", "higher is better"],
  davies_bouldin_ratio: ["DB Ratio", "higher is better"],
};

function showMetrics() {
  try {
    const m = runner.compute_metrics();
    let html = '<div class="metrics-grid">';
    for (const [key, [label, hint]] of Object.entries(METRIC_LABELS)) {
      if (m[key] !== undefined) {
        html += `<div class="metric-row">
          <span class="metric-name" title="${hint}">${label}</span>
          <span class="metric-value">${m[key].toFixed(4)}</span>
        </div>`;
      }
    }
    html += "</div>";
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
        showMetrics();
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
