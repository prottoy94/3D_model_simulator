/**
 * Phase 1 acceptance tests against the six reference notebooks
 * (docs/07-NOTEBOOK-PATTERNS.md). The notebook fixtures are git-ignored because
 * they hold private source code, so each test skips itself when its fixture is
 * missing. Recreate them with `pnpm make-fixture` (see fixtures/README.md).
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { Parser } from "web-tree-sitter";
import { loadPythonParserForNode } from "../python/test-utils";
import { analyzeNotebook, type NotebookAnalysis } from "./analyze-notebook";
import { parseNotebook } from "./parse";

type Expectation = {
  frameworks: string[];
  modelCells: number[];
  /** cell index → total params printed by that cell's summary. */
  summaryTotals: Record<number, number>;
  constants?: Record<string, unknown>;
  classCounts?: number[];
};

const expectations: Record<string, Expectation> = {
  "n1-alexnet-vgg": {
    frameworks: ["keras", "tensorflow"],
    modelCells: [14, 26],
    summaryTotals: { 16: 46781192, 28: 14981448 },
    constants: { IMG_HEIGHT: 224, IMG_SIZE: [224, 224], EPOCHS: 40 },
    classCounts: [8],
  },
  "n2-efficientnet": {
    frameworks: ["keras", "tensorflow", "sklearn"],
    modelCells: [16, 17],
    summaryTotals: { 19: 4214571, 23: 4214571 },
    constants: { FINE_TUNE_LAYERS: 30 },
    classCounts: [8],
  },
  "n3-mobilenetv2": {
    frameworks: ["keras", "tensorflow"],
    modelCells: [15, 16],
    summaryTotals: { 18: 2422984 },
    classCounts: [8],
  },
  "n4-vit": {
    frameworks: ["keras", "tensorflow", "sklearn"],
    modelCells: [16, 40, 41],
    summaryTotals: { 18: 404040 },
    constants: {
      PATCH_SIZE: 16,
      NUM_PATCHES: 196,
      PROJECTION_DIM: 64,
      NUM_HEADS: 4,
      TRANSFORMER_LAYERS: 4,
      MLP_DIM: 128,
      FINE_TUNE_BLOCKS: 2,
    },
    classCounts: [8],
  },
  "n5-rcnn": {
    frameworks: ["keras", "tensorflow"],
    modelCells: [26, 38, 43, 58, 66],
    summaryTotals: { 26: 1143113, 43: 897869, 58: 246637 },
    constants: { CANVAS_SIZE: 224, ROI_SIZE: 64, FEATURE_STRIDE: 8, TOTAL_ANCHORS: 7056 },
    classCounts: [8],
  },
  "n6-firenet": {
    frameworks: ["pytorch", "sklearn", "ultralytics"],
    modelCells: [9],
    summaryTotals: {},
  },
};

let parser: Parser;
beforeAll(async () => {
  parser = await loadPythonParserForNode();
});

for (const [name, expected] of Object.entries(expectations)) {
  const file = path.join(process.cwd(), "fixtures", "notebooks", `${name}.ipynb`);

  describe.skipIf(!existsSync(file))(`reference notebook ${name}`, () => {
    let analysis: NotebookAnalysis;
    let elapsedMs: number;

    beforeAll(() => {
      const text = readFileSync(file, "utf8");
      const started = performance.now();
      analysis = analyzeNotebook(parseNotebook(text), parser);
      elapsedMs = performance.now() - started;
    });

    it("tags exactly the model cells", () => {
      expect(analysis.modelCells).toEqual(expected.modelCells);
    });

    it("detects the frameworks", () => {
      expect(analysis.frameworks).toEqual(expected.frameworks);
    });

    it("reads the saved summaries", () => {
      const totals: Record<number, number> = {};
      for (const cell of analysis.cells) {
        const total = cell.summaries[0]?.totals.total;
        if (typeof total === "number") totals[cell.index] = total;
      }
      expect(totals).toMatchObject(expected.summaryTotals);
    });

    it("resolves constants and class counts", () => {
      expect(analysis.constants).toMatchObject(expected.constants ?? {});
      if (expected.classCounts) expect(analysis.classCounts).toEqual(expected.classCounts);
    });

    it("parses every code cell without syntax errors", () => {
      const broken = analysis.cells
        .filter((cell) => cell.analysis?.hasSyntaxError)
        .map((cell) => cell.index);
      expect(broken).toEqual([]);
    });

    it("parses and analyzes in well under a second", () => {
      expect(elapsedMs).toBeLessThan(1000);
    });
  });
}

describe("FireNet module hierarchy (n6)", () => {
  const file = path.join(process.cwd(), "fixtures", "notebooks", "n6-firenet.ipynb");
  it.skipIf(!existsSync(file))("finds the nn.Module classes with their line ranges", () => {
    const analysis = analyzeNotebook(parseNotebook(readFileSync(file, "utf8")), parser);
    const modules = analysis.symbols.filter(
      (symbol) => symbol.kind === "class" && symbol.bases.includes("nn.Module"),
    );
    expect(modules.map((symbol) => symbol.name)).toEqual([
      "ChannelAttention",
      "SpatialAttention",
      "CascadedAttention",
      "MultiScaleAttention",
      "ClassificationHead",
      "DetectionHead",
      "FireNet",
      "ResNetClassifier",
    ]);
    const fireNet = modules.find((symbol) => symbol.name === "FireNet");
    expect(fireNet?.cell).toBe(9);
    expect(fireNet?.methods).toEqual(
      expect.arrayContaining(["__init__", "encode", "classify", "detect", "forward"]),
    );
  });
});
