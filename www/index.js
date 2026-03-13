import {
  EmbeddingRunner,
  generate_sample_data,
  get_dataset_names,
  get_default_config,
  default as init,
} from "fitting-web";

// State
let dataSource = "random";
let runner = null;
let animationId = null;
let mnistCache = null;

// DOM refs
let canvas, status, lossDisplay, fpsDisplay, runBtn, stopBtn, stepBtn;
let progressContainer, progressBar, metricsPanel, metricsContent;
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
  document.getElementById("ee_iterations").value = d.early_exaggeration_iterations;
  document.getElementById("centering_weight").value = d.centering_weight;
  document.getElementById("scaling_loss").value = d.scaling_loss;
}

function main() {
  canvas = document.getElementById("canvas");
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

  setupCanvas();
  setupUI();

  status.textContent = "WebAssembly loaded! Click Run to start.";
}

function setupUI() {
  document
    .getElementById("btn-random")
    .addEventListener("click", () => setDataSource("random"));
  document
    .getElementById("btn-synthetic")
    .addEventListener("click", () => setDataSource("synthetic"));
  document
    .getElementById("btn-mnist")
    .addEventListener("click", () => setDataSource("mnist"));
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
      runner.render();
    }
  });
}

function setupCanvas() {
  const container = canvas.parentNode;
  // Use the smaller of available width/height to keep it square and fitting
  const maxW = container.clientWidth - 32;
  const maxH = window.innerHeight - 100;
  const size = Math.min(maxW, maxH, 600);
  canvas.width = size;
  canvas.height = size;
  canvas.style.width = size + "px";
  canvas.style.height = size + "px";
}

function setDataSource(source) {
  dataSource = source;
  document
    .getElementById("btn-random")
    .classList.toggle("active", source === "random");
  document
    .getElementById("btn-synthetic")
    .classList.toggle("active", source === "synthetic");
  document
    .getElementById("btn-mnist")
    .classList.toggle("active", source === "mnist");
  document.getElementById("random-controls").style.display =
    source === "random" ? "block" : "none";
  document.getElementById("synthetic-controls").style.display =
    source === "synthetic" ? "block" : "none";
  document.getElementById("mnist-controls").style.display =
    source === "mnist" ? "block" : "none";
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
    projection: document.getElementById("projection").value,
  };
}

// --- MNIST loading ---

async function fetchMnistRaw() {
  const [imagesResp, labelsResp] = await Promise.all([
    fetch("data/t10k-images-idx3-ubyte"),
    fetch("data/t10k-labels-idx1-ubyte"),
  ]);

  if (!imagesResp.ok || !labelsResp.ok) {
    throw new Error("MNIST data files not found.");
  }

  const [imagesBuf, labelsBuf] = await Promise.all([
    imagesResp.arrayBuffer(),
    labelsResp.arrayBuffer(),
  ]);

  const imagesView = new DataView(imagesBuf);
  const nImages = imagesView.getUint32(4);
  const rows = imagesView.getUint32(8);
  const cols = imagesView.getUint32(12);
  const nFeatures = rows * cols; // 784
  const imageBytes = new Uint8Array(imagesBuf, 16);
  const labelBytes = new Uint8Array(labelsBuf, 8);

  return { imageBytes, labelBytes, nImages, nFeatures };
}

async function loadMnist(nPoints) {
  if (!mnistCache) {
    mnistCache = await fetchMnistRaw();
  }

  const { imageBytes, labelBytes, nImages, nFeatures } = mnistCache;
  const nSamples = Math.min(nPoints, nImages);

  // Subsample evenly across the dataset
  const step = Math.floor(nImages / nSamples);
  const data = new Float64Array(nSamples * nFeatures);
  const labels = new Uint32Array(nSamples);

  for (let i = 0; i < nSamples; i++) {
    const srcIdx = i * step;
    const srcOffset = srcIdx * nFeatures;
    const dstOffset = i * nFeatures;
    for (let j = 0; j < nFeatures; j++) {
      data[dstOffset + j] = imageBytes[srcOffset + j] / 255.0;
    }
    labels[i] = labelBytes[srcIdx];
  }

  return { data, labels, nPoints: nSamples, nFeatures };
}

// --- Runner creation ---

async function createRunner() {
  if (runner !== null) {
    runner.free();
    runner = null;
  }

  const p = getParams();

  if (dataSource === "mnist") {
    const nPoints = parseInt(
      document.getElementById("mnist_n_points").value,
    );
    status.textContent = "Loading MNIST data...";
    const mnist = await loadMnist(nPoints);
    runner = EmbeddingRunner.from_data_with_labels(
      "canvas",
      mnist.data,
      mnist.labels,
      mnist.nPoints,
      mnist.nFeatures,
      p.curvature,
      p.iterations,
      p.perplexity,
      p.lr,
      p.eeFactor,
      p.eeIterations,
      p.centeringWeight,
      p.scalingLoss,
      p.projection,
    );
  } else if (dataSource === "synthetic") {
    const dataset = document.getElementById("dataset").value;
    const nPoints = parseInt(
      document.getElementById("synth_n_points").value,
    );
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
      p.projection,
    );
  } else {
    const nPoints = parseInt(document.getElementById("n_points").value);
    const nFeatures = parseInt(
      document.getElementById("n_features").value,
    );
    const data = generate_sample_data(nPoints, nFeatures, 42);
    runner = new EmbeddingRunner(
      "canvas",
      data,
      nPoints,
      nFeatures,
      p.curvature,
      p.iterations,
      p.perplexity,
      p.lr,
      p.eeFactor,
      p.eeIterations,
      p.centeringWeight,
      p.scalingLoss,
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
      runner.render();
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
    runner.render();
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
      fpsSmoothed = fpsSmoothed === 0 ? instantFps : fpsSmoothed * 0.9 + instantFps * 0.1;
      fpsDisplay.textContent = `${Math.round(fpsSmoothed)} FPS`;
    }

    const hasMore = runner.step(1);
    runner.render();
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
