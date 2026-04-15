/**
 * Pure data-parsing helpers — no fetch, no DOM, testable in Vitest.
 *
 * Each function takes already-fetched raw data and returns a plain JS object
 * with { data, labels, nPoints, nFeatures } (or { distances, labels, nPoints }
 * for the distance-based path).
 */

// ---------------------------------------------------------------------------
// IDX (MNIST / Fashion-MNIST)
// ---------------------------------------------------------------------------

const IDX_MAGIC_IMAGES = 0x00000803; // 2051
const IDX_MAGIC_LABELS = 0x00000801; // 2049

/**
 * Parse an IDX3 image buffer and IDX1 label buffer.
 *
 * Throws a descriptive error if the magic numbers are wrong (e.g., the files
 * are gzip-compressed — they must be decompressed before serving).
 *
 * @param {ArrayBuffer} imagesBuf
 * @param {ArrayBuffer} labelsBuf
 * @returns {{ imageBytes: Uint8Array, labelBytes: Uint8Array, nImages: number, nFeatures: number }}
 */
export function parseIdxBuffers(imagesBuf, labelsBuf) {
  const imagesView = new DataView(imagesBuf);
  const labelsView = new DataView(labelsBuf);

  const imagesMagic = imagesView.getUint32(0); // big-endian
  if (imagesMagic !== IDX_MAGIC_IMAGES) {
    throw new Error(
      `Invalid IDX image magic: expected 0x${IDX_MAGIC_IMAGES.toString(16).padStart(8, "0")}, ` +
        `got 0x${imagesMagic.toString(16).padStart(8, "0")}. ` +
        `The file may be gzip-compressed — place the raw .idx3-ubyte file (not .gz) in the data directory.`,
    );
  }

  const labelsMagic = labelsView.getUint32(0);
  if (labelsMagic !== IDX_MAGIC_LABELS) {
    throw new Error(
      `Invalid IDX label magic: expected 0x${IDX_MAGIC_LABELS.toString(16).padStart(8, "0")}, ` +
        `got 0x${labelsMagic.toString(16).padStart(8, "0")}. ` +
        `The file may be gzip-compressed — place the raw .idx1-ubyte file (not .gz) in the data directory.`,
    );
  }

  const nImages = imagesView.getUint32(4);
  const rows = imagesView.getUint32(8);
  const cols = imagesView.getUint32(12);
  const nFeatures = rows * cols;
  const imageBytes = new Uint8Array(imagesBuf, 16);
  const labelBytes = new Uint8Array(labelsBuf, 8);
  return { imageBytes, labelBytes, nImages, nFeatures };
}

/**
 * Evenly subsample `nPoints` items from an IDX raw parse result.
 *
 * @param {{ imageBytes, labelBytes, nImages, nFeatures }} raw
 * @param {number} nPoints
 * @returns {{ data: Float64Array, labels: Uint32Array, nPoints: number, nFeatures: number }}
 */
export function subsampleIdx({ imageBytes, labelBytes, nImages, nFeatures }, nPoints) {
  const nSamples = Math.min(nPoints, nImages);
  const step = Math.max(1, Math.floor(nImages / nSamples));
  const data = new Float64Array(nSamples * nFeatures);
  const labels = new Uint32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const srcIdx = i * step;
    const srcOff = srcIdx * nFeatures;
    const dstOff = i * nFeatures;
    for (let j = 0; j < nFeatures; j++) {
      data[dstOff + j] = imageBytes[srcOff + j] / 255.0;
    }
    labels[i] = labelBytes[srcIdx];
  }
  return { data, labels, nPoints: nSamples, nFeatures };
}

// ---------------------------------------------------------------------------
// PBMC PCA (TSV / CSV / whitespace-separated)
// ---------------------------------------------------------------------------

/**
 * Detect the field separator in a text line.
 * Checks for tab, then comma, then falls back to whitespace.
 *
 * @param {string} line
 * @returns {'\t' | ',' | 'whitespace'}
 */
export function detectSeparator(line) {
  if (line.includes("\t")) return "\t";
  if (line.includes(",")) return ",";
  return "whitespace";
}

/**
 * Split a line using a separator returned by `detectSeparator`.
 *
 * @param {string} line
 * @param {'\t' | ',' | 'whitespace'} sep
 * @returns {string[]}
 */
export function splitLine(line, sep) {
  if (sep === "whitespace") return line.trim().split(/\s+/);
  return line.split(sep);
}

/**
 * Parse a PBMC PCA text file.
 *
 * Handles:
 * - Tab, comma, or whitespace-separated fields (auto-detected).
 * - Optional header row (first field is non-numeric → skip).
 * - Optional label column (first field of data rows is non-numeric → use as class label).
 * - Windows (\r\n) and Unix (\n) line endings.
 * - If there are more than MAX_LABEL_CLASSES unique string labels (e.g., unique barcodes),
 *   the label column is treated as an index and all points receive label 0.
 *
 * @param {string} text       Full file contents.
 * @param {number} nPoints    Maximum number of samples to return.
 * @returns {{ data: Float64Array, labels: Uint32Array, nPoints: number, nFeatures: number }}
 */
export function parsePbmcText(text, nPoints) {
  const MAX_LABEL_CLASSES = 50; // above this, treat column as index not label

  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#"));

  if (lines.length === 0) throw new Error("PBMC PCA file is empty");

  // Detect separator from the first line
  const sep = detectSeparator(lines[0]);

  // Detect optional header row.
  // A header row has ALL fields non-numeric (e.g. "PC1\tPC2\tPC3").
  // A data row with a string label has only the first field non-numeric (e.g. "T cell\t1.0\t2.0").
  const firstFields = splitLine(lines[0], sep);
  const hasHeader =
    firstFields.length > 1
      ? isNaN(parseFloat(firstFields[0])) && isNaN(parseFloat(firstFields[1]))
      : isNaN(parseFloat(firstFields[0]));
  const dataLines = hasHeader ? lines.slice(1) : lines;

  if (dataLines.length === 0)
    throw new Error("PBMC PCA file has no data rows after header");

  // Detect optional label column: first field of first data row is non-numeric
  const firstDataFields = splitLine(dataLines[0], sep);
  const hasLabelCol = isNaN(parseFloat(firstDataFields[0]));
  const featureStart = hasLabelCol ? 1 : 0;

  const nFeatures = firstDataFields.length - featureStart;
  if (nFeatures === 0)
    throw new Error(
      "PBMC PCA file: no feature columns detected. " +
        `Separator detected as '${sep === "whitespace" ? "whitespace" : sep}'. ` +
        "Ensure the file is tab-, comma-, or space-separated.",
    );

  const nSamples = Math.min(nPoints, dataLines.length);
  const step = Math.max(1, Math.floor(dataLines.length / nSamples));

  const data = new Float64Array(nSamples * nFeatures);
  const rawLabels = [];

  for (let i = 0; i < nSamples; i++) {
    const fields = splitLine(dataLines[i * step], sep);
    if (hasLabelCol) rawLabels.push(fields[0].trim());
    for (let j = 0; j < nFeatures; j++) {
      data[i * nFeatures + j] = parseFloat(fields[featureStart + j]);
    }
  }

  const labels = new Uint32Array(nSamples);
  let labelNames = null;
  if (hasLabelCol && rawLabels.length > 0) {
    const uniqueLabels = [...new Set(rawLabels)].sort();
    if (uniqueLabels.length <= MAX_LABEL_CLASSES) {
      const labelMap = new Map(uniqueLabels.map((l, idx) => [l, idx]));
      for (let i = 0; i < nSamples; i++) labels[i] = labelMap.get(rawLabels[i]) ?? 0;
      labelNames = uniqueLabels; // string name for each integer label value
    }
    // else: too many unique values (e.g. barcodes) — leave all labels as 0
  }

  return { data, labels, labelNames, nPoints: nSamples, nFeatures };
}

// ---------------------------------------------------------------------------
// WordNet mammals (edge list → BFS distance matrix)
// ---------------------------------------------------------------------------

/**
 * Parse WordNet edge list and optional labels/names text files into a
 * pairwise distance matrix.
 *
 * @param {string} edgesText   Tab-separated parent_id\tchild_id lines.
 * @param {string|null} labelsText  One integer label per line (node i = line i), or null.
 * @param {string|null} namesText   One short name per line (node i = line i), or null.
 * @param {number} nPoints     Maximum number of nodes to include (BFS from root).
 * @returns {{ distances: Float64Array, labels: Uint32Array, names: string[], nPoints: number }}
 */
export function parseWordnetEdges(edgesText, labelsText, namesText, nPoints) {
  // Parse edge list
  const edges = [];
  let maxId = 0;
  for (const line of edgesText.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const parts = trimmed.split("\t");
    const p = parseInt(parts[0]);
    const c = parseInt(parts[1]);
    if (!isNaN(p) && !isNaN(c)) {
      edges.push([p, c]);
      maxId = Math.max(maxId, p, c);
    }
  }
  const nNodesFull = maxId + 1;

  // Build undirected adjacency list
  const adj = Array.from({ length: nNodesFull }, () => []);
  for (const [p, c] of edges) {
    adj[p].push(c);
    adj[c].push(p);
  }

  // BFS from root (node 0) to select up to nPoints nodes in BFS order
  const n = Math.min(nPoints, nNodesFull);
  const bfsOrder = [];
  const visited = new Uint8Array(nNodesFull);
  const queue = [0];
  visited[0] = 1;
  let head = 0;
  while (head < queue.length && bfsOrder.length < n) {
    const node = queue[head++];
    bfsOrder.push(node);
    for (const nb of adj[node]) {
      if (!visited[nb]) {
        visited[nb] = 1;
        queue.push(nb);
      }
    }
  }
  const nActual = bfsOrder.length;

  // Map original node IDs → compact indices
  const idToIdx = new Int32Array(nNodesFull).fill(-1);
  for (let i = 0; i < nActual; i++) idToIdx[bfsOrder[i]] = i;

  // Build compact adjacency list
  const compactAdj = Array.from({ length: nActual }, () => []);
  for (const [p, c] of edges) {
    const pi = idToIdx[p];
    const ci = idToIdx[c];
    if (pi >= 0 && ci >= 0) {
      compactAdj[pi].push(ci);
      compactAdj[ci].push(pi);
    }
  }

  // All-pairs BFS distances
  const INF = nActual * 2;
  const distances = new Float64Array(nActual * nActual).fill(INF);
  for (let src = 0; src < nActual; src++) {
    distances[src * nActual + src] = 0;
    const q = [src];
    let h = 0;
    while (h < q.length) {
      const u = q[h++];
      const du = distances[src * nActual + u];
      for (const v of compactAdj[u]) {
        if (distances[src * nActual + v] === INF) {
          distances[src * nActual + v] = du + 1;
          q.push(v);
        }
      }
    }
  }

  // Labels from file
  let labels;
  if (labelsText) {
    const rawLabels = labelsText
      .split(/\r?\n/)
      .map((l) => parseInt(l.trim()))
      .filter((v) => !isNaN(v));
    labels = new Uint32Array(nActual);
    for (let i = 0; i < nActual; i++) {
      labels[i] = rawLabels[bfsOrder[i]] ?? 0;
    }
  } else {
    labels = new Uint32Array(nActual);
  }

  // Names from file (mapped to compact BFS order)
  let names;
  if (namesText) {
    const rawNames = namesText.split(/\r?\n/).map((l) => l.trim());
    names = Array.from({ length: nActual }, (_, i) => rawNames[bfsOrder[i]] ?? "");
  } else {
    names = Array(nActual).fill("");
  }

  // Collect deduplicated edges in compact index space (i < j to avoid duplicates)
  const compactEdges = [];
  for (let i = 0; i < nActual; i++) {
    for (const j of compactAdj[i]) {
      if (i < j) compactEdges.push([i, j]);
    }
  }

  return { distances, labels, names, edges: compactEdges, nPoints: nActual };
}
