import { describe, it, expect } from "vitest";
import {
  parseIdxBuffers,
  subsampleIdx,
  detectSeparator,
  splitLine,
  parsePbmcText,
  parseWordnetEdges,
} from "../dataLoaders.js";

// ---------------------------------------------------------------------------
// Helpers to build synthetic test data
// ---------------------------------------------------------------------------

/** Build a minimal IDX image buffer (big-endian). */
function makeIdxImageBuf(nImages, rows, cols, pixelValue = 128) {
  const headerSize = 16; // 4 × uint32
  const dataSize = nImages * rows * cols;
  const buf = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buf);
  view.setUint32(0, 0x00000803); // magic
  view.setUint32(4, nImages);
  view.setUint32(8, rows);
  view.setUint32(12, cols);
  const bytes = new Uint8Array(buf, headerSize);
  bytes.fill(pixelValue);
  return buf;
}

/** Build a minimal IDX label buffer (big-endian). */
function makeIdxLabelBuf(labels) {
  const headerSize = 8; // 2 × uint32
  const buf = new ArrayBuffer(headerSize + labels.length);
  const view = new DataView(buf);
  view.setUint32(0, 0x00000801); // magic
  view.setUint32(4, labels.length);
  const bytes = new Uint8Array(buf, headerSize);
  bytes.set(labels);
  return buf;
}

// ---------------------------------------------------------------------------
// parseIdxBuffers
// ---------------------------------------------------------------------------

describe("parseIdxBuffers", () => {
  it("parses a valid IDX image + label buffer", () => {
    const nImages = 5;
    const rows = 2;
    const cols = 3;
    const imgBuf = makeIdxImageBuf(nImages, rows, cols, 255);
    const lblBuf = makeIdxLabelBuf(new Uint8Array([0, 1, 2, 3, 4]));

    const result = parseIdxBuffers(imgBuf, lblBuf);
    expect(result.nImages).toBe(nImages);
    expect(result.nFeatures).toBe(rows * cols);
    expect(result.imageBytes.length).toBe(nImages * rows * cols);
    expect(result.labelBytes[2]).toBe(2);
  });

  it("throws when image magic is wrong (e.g., gzip header 0x1f8b0800)", () => {
    const buf = new ArrayBuffer(20);
    const view = new DataView(buf);
    view.setUint32(0, 0x1f8b0800); // gzip magic
    const lblBuf = makeIdxLabelBuf(new Uint8Array([0]));

    expect(() => parseIdxBuffers(buf, lblBuf)).toThrow(/Invalid IDX image magic/);
    expect(() => parseIdxBuffers(buf, lblBuf)).toThrow(/gzip/i);
  });

  it("throws when label magic is wrong", () => {
    const imgBuf = makeIdxImageBuf(1, 2, 2, 0);
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, 0xdeadbeef); // wrong magic

    expect(() => parseIdxBuffers(imgBuf, buf)).toThrow(/Invalid IDX label magic/);
  });

  it("pixel bytes start right after the 16-byte header", () => {
    const imgBuf = makeIdxImageBuf(1, 1, 1, 42);
    const lblBuf = makeIdxLabelBuf(new Uint8Array([7]));
    const { imageBytes, labelBytes } = parseIdxBuffers(imgBuf, lblBuf);
    expect(imageBytes[0]).toBe(42);
    expect(labelBytes[0]).toBe(7);
  });
});

// ---------------------------------------------------------------------------
// subsampleIdx
// ---------------------------------------------------------------------------

describe("subsampleIdx", () => {
  it("returns all items when nPoints >= nImages", () => {
    const imgBuf = makeIdxImageBuf(3, 2, 2, 255);
    const lblBuf = makeIdxLabelBuf(new Uint8Array([0, 1, 2]));
    const raw = parseIdxBuffers(imgBuf, lblBuf);
    const result = subsampleIdx(raw, 10);
    expect(result.nPoints).toBe(3);
    expect(result.data.length).toBe(3 * 4);
  });

  it("subsamples when nPoints < nImages", () => {
    const imgBuf = makeIdxImageBuf(100, 4, 4, 200);
    const lblBuf = makeIdxLabelBuf(new Uint8Array(100).fill(5));
    const raw = parseIdxBuffers(imgBuf, lblBuf);
    const result = subsampleIdx(raw, 10);
    expect(result.nPoints).toBe(10);
    expect(result.data.length).toBe(10 * 16);
  });

  it("normalizes pixel values to [0, 1]", () => {
    const imgBuf = makeIdxImageBuf(1, 1, 1, 255);
    const lblBuf = makeIdxLabelBuf(new Uint8Array([0]));
    const raw = parseIdxBuffers(imgBuf, lblBuf);
    const { data } = subsampleIdx(raw, 1);
    expect(data[0]).toBeCloseTo(1.0, 5);
  });

  it("returns 0-normalized for black pixels", () => {
    const imgBuf = makeIdxImageBuf(1, 1, 1, 0);
    const lblBuf = makeIdxLabelBuf(new Uint8Array([0]));
    const raw = parseIdxBuffers(imgBuf, lblBuf);
    const { data } = subsampleIdx(raw, 1);
    expect(data[0]).toBe(0.0);
  });
});

// ---------------------------------------------------------------------------
// detectSeparator / splitLine
// ---------------------------------------------------------------------------

describe("detectSeparator", () => {
  it("detects tab", () => {
    expect(detectSeparator("foo\tbar\tbaz")).toBe("\t");
  });

  it("detects comma when no tab present", () => {
    expect(detectSeparator("foo,bar,baz")).toBe(",");
  });

  it("falls back to whitespace", () => {
    expect(detectSeparator("foo bar baz")).toBe("whitespace");
  });

  it("prefers tab over comma when both present", () => {
    expect(detectSeparator("a\tb,c")).toBe("\t");
  });
});

describe("splitLine", () => {
  it("splits by tab", () => {
    expect(splitLine("a\tb\tc", "\t")).toEqual(["a", "b", "c"]);
  });

  it("splits by comma", () => {
    expect(splitLine("a,b,c", ",")).toEqual(["a", "b", "c"]);
  });

  it("splits by whitespace and trims", () => {
    expect(splitLine("  a   b  c  ", "whitespace")).toEqual(["a", "b", "c"]);
  });
});

// ---------------------------------------------------------------------------
// parsePbmcText
// ---------------------------------------------------------------------------

describe("parsePbmcText", () => {
  it("parses a simple tab-separated file without header or labels", () => {
    const text = ["1.0\t2.0\t3.0", "4.0\t5.0\t6.0", "7.0\t8.0\t9.0"].join("\n");
    const result = parsePbmcText(text, 10);
    expect(result.nPoints).toBe(3);
    expect(result.nFeatures).toBe(3);
    expect(result.data[0]).toBeCloseTo(1.0);
    expect(result.data[5]).toBeCloseTo(6.0);
  });

  it("detects and skips a header row", () => {
    const text = ["PC1\tPC2\tPC3", "1.0\t2.0\t3.0", "4.0\t5.0\t6.0"].join("\n");
    const result = parsePbmcText(text, 10);
    expect(result.nPoints).toBe(2);
    expect(result.nFeatures).toBe(3);
    expect(result.data[0]).toBeCloseTo(1.0);
  });

  it("detects a string label column and maps labels to integers", () => {
    const text = [
      "T cell\t1.0\t2.0",
      "B cell\t3.0\t4.0",
      "T cell\t5.0\t6.0",
    ].join("\n");
    const result = parsePbmcText(text, 10);
    expect(result.nFeatures).toBe(2);
    expect(result.nPoints).toBe(3);
    // "B cell" sorts before "T cell" → B cell=0, T cell=1
    expect(result.labels[0]).toBe(1); // T cell
    expect(result.labels[1]).toBe(0); // B cell
    expect(result.labels[2]).toBe(1); // T cell
    expect(result.labelNames).toEqual(["B cell", "T cell"]);
  });

  it("returns null labelNames when there is no label column", () => {
    const text = ["1.0\t2.0", "3.0\t4.0"].join("\n");
    const result = parsePbmcText(text, 10);
    expect(result.labelNames).toBeNull();
  });

  it("returns null labelNames when too many unique values", () => {
    const rows = Array.from({ length: 60 }, (_, i) => `BARCODE${i}\t1.0\t2.0`);
    const result = parsePbmcText(rows.join("\n"), 60);
    expect(result.labelNames).toBeNull();
  });

  it("handles comma-separated files", () => {
    const text = ["1.0,2.0,3.0", "4.0,5.0,6.0"].join("\n");
    const result = parsePbmcText(text, 10);
    expect(result.nFeatures).toBe(3);
    expect(result.data[3]).toBeCloseTo(4.0);
  });

  it("handles whitespace-separated files", () => {
    const text = ["1.0 2.0 3.0", "4.0 5.0 6.0"].join("\n");
    const result = parsePbmcText(text, 10);
    expect(result.nFeatures).toBe(3);
    expect(result.data[0]).toBeCloseTo(1.0);
  });

  it("handles Windows line endings (\\r\\n)", () => {
    const text = "1.0\t2.0\r\n3.0\t4.0\r\n";
    const result = parsePbmcText(text, 10);
    expect(result.nPoints).toBe(2);
    expect(result.nFeatures).toBe(2);
  });

  it("ignores comment lines starting with #", () => {
    const text = ["# header comment", "1.0\t2.0", "3.0\t4.0"].join("\n");
    const result = parsePbmcText(text, 10);
    expect(result.nPoints).toBe(2);
  });

  it("subsamples when nPoints < total rows", () => {
    const rows = Array.from({ length: 100 }, (_, i) => `${i}.0\t${i + 1}.0`);
    const text = rows.join("\n");
    const result = parsePbmcText(text, 10);
    expect(result.nPoints).toBe(10);
  });

  it("treats label column as index (all labels=0) when too many unique values", () => {
    // Simulate a file with barcodes (all unique) as first column
    const rows = Array.from(
      { length: 60 },
      (_, i) => `BARCODE${String(i).padStart(4, "0")}-1\t1.0\t2.0`,
    );
    const text = rows.join("\n");
    const result = parsePbmcText(text, 60);
    expect(result.nFeatures).toBe(2);
    expect(result.labels.every((l) => l === 0)).toBe(true);
  });

  it("throws on empty input", () => {
    expect(() => parsePbmcText("", 10)).toThrow(/empty/i);
    expect(() => parsePbmcText("   \n  \n", 10)).toThrow(/empty/i);
  });

  it("throws when no feature columns can be detected", () => {
    // Only header, no data rows
    expect(() => parsePbmcText("PC1\tPC2\tPC3", 10)).toThrow(/no data rows/i);
  });
});

// ---------------------------------------------------------------------------
// parseWordnetEdges
// ---------------------------------------------------------------------------

describe("parseWordnetEdges", () => {
  // Simple 4-node tree:  0─1─2, 0─3
  const EDGES = "0\t1\n1\t2\n0\t3\n";
  const LABELS = "0\n0\n1\n2\n"; // node 0→0, 1→0, 2→1, 3→2
  const NAMES = "mammal\ncanine\nwolf\nwhale\n";

  it("includes all nodes when nPoints >= tree size", () => {
    const result = parseWordnetEdges(EDGES, null, null, 100);
    expect(result.nPoints).toBe(4);
  });

  it("limits to nPoints nodes in BFS order", () => {
    const result = parseWordnetEdges(EDGES, null, null, 2);
    expect(result.nPoints).toBe(2); // root (0) and first BFS child
  });

  it("distance matrix is symmetric and zero on diagonal", () => {
    const { distances, nPoints } = parseWordnetEdges(EDGES, null, null, 100);
    for (let i = 0; i < nPoints; i++) {
      expect(distances[i * nPoints + i]).toBe(0);
      for (let j = 0; j < nPoints; j++) {
        expect(distances[i * nPoints + j]).toBe(distances[j * nPoints + i]);
      }
    }
  });

  it("computes correct shortest-path distances on a chain 0-1-2", () => {
    const chainEdges = "0\t1\n1\t2\n";
    const { distances, nPoints } = parseWordnetEdges(chainEdges, null, null, 100);
    expect(nPoints).toBe(3);
    expect(distances[0 * 3 + 2]).toBe(2); // 0→2 takes 2 hops
    expect(distances[0 * 3 + 1]).toBe(1); // 0→1 takes 1 hop
  });

  it("loads integer labels from labels text", () => {
    const { labels } = parseWordnetEdges(EDGES, LABELS, null, 100);
    // BFS order from root 0: 0,1,3,2 (or 0,3,1,2 depending on adj order)
    // We only check that all labels are from {0,1,2}
    expect(labels.every((l) => l >= 0 && l <= 2)).toBe(true);
  });

  it("returns empty string names when no names file", () => {
    const { names } = parseWordnetEdges(EDGES, null, null, 100);
    expect(names.every((n) => n === "")).toBe(true);
  });

  it("loads names from names text in BFS order", () => {
    const { names, nPoints } = parseWordnetEdges(EDGES, null, NAMES, 100);
    expect(names.length).toBe(nPoints);
    // Root (node 0) should map to "mammal"
    expect(names[0]).toBe("mammal");
  });

  it("handles comment lines and blank lines in edge file", () => {
    const edgesWithComments = "# comment\n\n0\t1\n\n1\t2\n";
    const result = parseWordnetEdges(edgesWithComments, null, null, 100);
    expect(result.nPoints).toBe(3);
  });

  it("handles Windows line endings", () => {
    const edges = "0\t1\r\n1\t2\r\n";
    const result = parseWordnetEdges(edges, null, null, 100);
    expect(result.nPoints).toBe(3);
  });

  it("returns deduplicated edges with i < j", () => {
    // Tree: 0-1-2, 0-3  →  3 edges
    const { edges } = parseWordnetEdges(EDGES, null, null, 100);
    expect(edges.length).toBe(3);
    // Every edge should have src < dst (no duplicates)
    for (const [a, b] of edges) {
      expect(a).toBeLessThan(b);
    }
    // All indices are within bounds
    const { nPoints } = parseWordnetEdges(EDGES, null, null, 100);
    for (const [a, b] of edges) {
      expect(a).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThan(nPoints);
    }
  });

  it("edges only include nodes within the BFS-limited subgraph", () => {
    // With nPoints=2 only root (0) and one child are included; only 1 edge
    const { edges } = parseWordnetEdges(EDGES, null, null, 2);
    expect(edges.length).toBe(1);
  });
});
