import {
  run_embedding_animated,
  run_synthetic_embedding,
  generate_sample_data,
  get_dataset_names,
  default as init,
} from "fitting-web";

// State
let dataSource = "random";

// DOM refs (set in main)
let canvas, status, lossDisplay, runBtn;

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
  window.addEventListener("resize", setupCanvas);
}

function setupCanvas() {
  const dpr = window.devicePixelRatio || 1.0;
  const aspectRatio = canvas.width / canvas.height;
  const size = canvas.parentNode.offsetWidth * 0.8;
  canvas.style.width = size + "px";
  canvas.style.height = size / aspectRatio + "px";
  canvas.width = size;
  canvas.height = size / aspectRatio;
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

function runEmbedding() {
  runBtn.disabled = true;
  status.textContent = "Running...";
  lossDisplay.textContent = "";

  const curvature = parseFloat(document.getElementById("curvature").value);
  const perplexity = parseFloat(document.getElementById("perplexity").value);
  const iterations = parseInt(document.getElementById("iterations").value);
  const lr = parseFloat(document.getElementById("lr").value);
  const renderEvery = parseInt(document.getElementById("render_every").value);

  // Use setTimeout so the UI paints "Running..." before we block
  setTimeout(() => {
    const t0 = performance.now();
    try {
      if (dataSource === "synthetic") {
        const dataset = document.getElementById("dataset").value;
        const nPoints = parseInt(
          document.getElementById("synth_n_points").value,
        );
        run_synthetic_embedding(
          "canvas",
          dataset,
          nPoints,
          curvature,
          iterations,
          perplexity,
          lr,
          renderEvery,
        );
      } else {
        const nPoints = parseInt(document.getElementById("n_points").value);
        const nFeatures = parseInt(
          document.getElementById("n_features").value,
        );
        const data = generate_sample_data(nPoints, nFeatures, 42);
        run_embedding_animated(
          "canvas",
          data,
          nPoints,
          nFeatures,
          curvature,
          iterations,
          perplexity,
          lr,
          renderEvery,
        );
      }
      const elapsed = Math.ceil(performance.now() - t0);
      status.textContent = `Done in ${elapsed}ms`;
    } catch (e) {
      status.textContent = "Error: " + e;
      console.error(e);
    }
    runBtn.disabled = false;
  }, 50);
}
