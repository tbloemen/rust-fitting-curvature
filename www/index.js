import {
  EmbeddingRunner,
  generate_sample_data,
  get_dataset_names,
  default as init,
} from "fitting-web";

// State
let dataSource = "random";
let runner = null;
let animationId = null;

// DOM refs
let canvas, status, lossDisplay, runBtn, stopBtn, stepBtn;

initialize();

async function initialize() {
  await init();
  main();
}

function main() {
  canvas = document.getElementById("canvas");
  status = document.getElementById("status");
  lossDisplay = document.getElementById("loss-display");
  runBtn = document.getElementById("run-btn");
  stopBtn = document.getElementById("stop-btn");
  stepBtn = document.getElementById("step-btn");

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
  document.getElementById("random-controls").style.display =
    source === "random" ? "block" : "none";
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
    projection: document.getElementById("projection").value,
  };
}

function createRunner() {
  if (runner !== null) {
    runner.free();
    runner = null;
  }

  const p = getParams();

  if (dataSource === "synthetic") {
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
  const loss = runner.loss();
  lossDisplay.textContent = `iter=${iter}  loss=${loss.toFixed(4)}`;
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
  status.textContent = "Reset. Click Run or Step to start.";
}

function stepEmbedding() {
  // Stop any running animation first
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  try {
    // Create runner if not yet initialized — show initial state (step 0)
    if (runner === null || runner.is_done()) {
      createRunner();
      runner.render();
      updateDisplay();
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
    }
  } catch (e) {
    status.textContent = "Error: " + e;
    console.error(e);
  }
}

function runEmbedding() {
  // Cancel any running animation
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  // Create a fresh runner only if none exists or it's done
  const resuming = runner !== null && !runner.is_done();
  if (!resuming) {
    try {
      createRunner();
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
  if (!resuming) lossDisplay.textContent = "";

  const t0 = performance.now();

  function animate() {
    const hasMore = runner.step(1);
    runner.render();
    updateDisplay();

    if (hasMore) {
      animationId = requestAnimationFrame(animate);
    } else {
      animationId = null;
      const elapsed = Math.ceil(performance.now() - t0);
      status.textContent = `Done in ${elapsed}ms (${runner.iteration()} iterations)`;
      runBtn.disabled = false;
      stopBtn.textContent = "Reset";
      stopBtn.disabled = false;
      stepBtn.disabled = true;
    }
  }

  animationId = requestAnimationFrame(animate);
}
