import { beforeAll, describe, expect, it } from "vitest";
import type { Parser } from "web-tree-sitter";
import { analyzeCell } from "./analyze";
import { formatStaticValue, type StaticValue } from "./evaluate";
import { sanitizePython } from "./sanitize";
import { loadPythonParserForNode } from "./test-utils";

let parser: Parser;
beforeAll(async () => {
  parser = await loadPythonParserForNode();
});

function analyze(source: string, scope = new Map<string, StaticValue>()) {
  const tree = parser.parse(sanitizePython(source));
  if (!tree) throw new Error("parse failed");
  return { analysis: analyzeCell(tree, 0, scope), scope };
}

describe("sanitizePython", () => {
  it("neutralizes IPython syntax and keeps every line number", () => {
    const source = [
      "!pip install --quiet keras-hub",
      "%matplotlib inline",
      "%%time",
      "import os",
      "if True:",
      "    !nvidia-smi",
      "files = !ls",
      "model?",
      "x = 1",
    ].join("\n");
    const clean = sanitizePython(source);
    expect(clean.split("\n")).toHaveLength(source.split("\n").length);
    expect(clean.split("\n")[0]).toBe("pass  # !pip install --quiet keras-hub");
    expect(clean.split("\n")[5]).toBe("    pass  # !nvidia-smi");
    expect(clean.split("\n")[6]).toBe("files = None  # !ls");
    expect(clean.split("\n")[8]).toBe("x = 1");
    expect(parser.parse(clean)?.rootNode.hasError).toBe(false);
  });

  it("leaves text inside multi-line strings alone", () => {
    const source = ['doc = """', "!not a command", "%neither", '"""', "!real"].join("\n");
    expect(sanitizePython(source).split("\n")).toEqual([
      'doc = """',
      "!not a command",
      "%neither",
      '"""',
      "pass  # !real",
    ]);
  });

  it("does not touch ordinary Python", () => {
    const source = "a = b % c\nok = not x != y\ns = '!'  # % comment";
    expect(sanitizePython(source)).toBe(source);
  });
});

describe("analyzeCell", () => {
  it("resolves constants like the mini-ViT config cell", () => {
    const { scope } = analyze(
      [
        "IMG_HEIGHT = 224",
        "IMG_WIDTH = 224",
        "IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)",
        "PATCH_SIZE = 16",
        "NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) ** 2   # 14 x 14 = 196",
        "RATE = 1e-4",
        'NAME = "vit" + "_scratch"',
      ].join("\n"),
    );
    expect(scope.get("NUM_PATCHES")).toBe(196);
    expect(scope.get("IMG_SIZE")).toEqual([224, 224]);
    expect(scope.get("RATE")).toBe(0.0001);
    expect(scope.get("NAME")).toBe("vit_scratch");
    expect(formatStaticValue(scope.get("IMG_SIZE")!)).toBe("(224, 224)");
  });

  it("resolves the Faster R-CNN anchor arithmetic", () => {
    const { scope } = analyze(
      [
        "CANVAS_SIZE = 224",
        "FEATURE_MAP_SIZE = 28",
        "FEATURE_STRIDE = CANVAS_SIZE / FEATURE_MAP_SIZE",
        "ANCHOR_SCALES = [32, 64, 128]",
        "ANCHOR_RATIOS = [0.5, 1.0, 2.0]",
        "NUM_ANCHORS_PER_LOCATION = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)",
        "TOTAL_ANCHORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_ANCHORS_PER_LOCATION",
      ].join("\n"),
    );
    expect(scope.get("FEATURE_STRIDE")).toBe(8);
    expect(scope.get("NUM_ANCHORS_PER_LOCATION")).toBe(9);
    expect(scope.get("TOTAL_ANCHORS")).toBe(7056);
  });

  it("keeps values unknown when it cannot be sure", () => {
    const { scope, analysis } = analyze(
      [
        "A = 1",
        "num_classes = len(class_names)",
        "if smoke:",
        "    IMG = 96",
        "else:",
        "    IMG = 224",
        "H, W = 224, 112",
        "A = load()",
      ].join("\n"),
    );
    expect(scope.has("num_classes")).toBe(false);
    expect(scope.has("IMG")).toBe(false);
    expect(scope.get("H")).toBe(224);
    expect(scope.get("W")).toBe(112);
    expect(scope.has("A")).toBe(false); // reassigned to an unknown value
    expect(analysis.defines.filter((symbol) => symbol.name === "IMG")).toHaveLength(2);
  });

  it("carries constants across cells through the shared scope", () => {
    const scope = new Map<string, StaticValue>();
    analyze("PROJECTION_DIM = 64", scope);
    analyze("MLP_DIM = PROJECTION_DIM * 2", scope);
    expect(scope.get("MLP_DIM")).toBe(128);
  });

  it("collects classes, functions, imports, calls and attribute writes", () => {
    const { analysis } = analyze(
      [
        "import torch.nn as nn",
        "import torchvision",
        "from tensorflow.keras.layers import Dense, Conv2D as C2",
        "class ChannelAttention(nn.Module):",
        "    def __init__(self, ch):",
        "        super().__init__()",
        "        self.mlp = nn.Sequential(nn.Conv2d(ch, 8, 1), nn.ReLU())",
        "    def forward(self, x):",
        "        return x",
        "@decorator",
        "def build(x):",
        "    return Dense(4)(x)",
        "base_model.trainable = False",
      ].join("\n"),
    );
    expect(analysis.hasSyntaxError).toBe(false);
    expect(analysis.importedModules).toEqual([
      "torch.nn",
      "torchvision",
      "tensorflow.keras.layers",
    ]);

    const byName = Object.fromEntries(analysis.defines.map((symbol) => [symbol.name, symbol]));
    expect(byName.nn).toMatchObject({ kind: "import", module: "torch.nn" });
    expect(byName.C2).toMatchObject({ kind: "import", module: "tensorflow.keras.layers.Conv2D" });
    expect(byName.ChannelAttention).toMatchObject({
      kind: "class",
      bases: ["nn.Module"],
      methods: ["__init__", "forward"],
      startLine: 4,
      endLine: 9,
    });
    expect(byName.build).toMatchObject({ kind: "function", startLine: 10, endLine: 12 });
    expect(analysis.calls).toEqual(
      expect.arrayContaining(["nn.Sequential", "nn.Conv2d", "nn.ReLU", "Dense"]),
    );
    expect(analysis.attributeWrites).toContain("base_model.trainable");
  });

  it("survives broken code", () => {
    const { analysis } = analyze("def broken(:\n    pass\nX = 3");
    expect(analysis.hasSyntaxError).toBe(true);
  });
});
