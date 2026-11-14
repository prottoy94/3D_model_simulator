import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { Parser } from "web-tree-sitter";
import { buildRuntimeIR } from "../ir/build";
import type { RuntimeIR } from "../ir/runtime";
import { analyzeNotebook } from "../notebook/analyze-notebook";
import { parseNotebook } from "../notebook/parse";
import { loadPythonParserForNode } from "../python/test-utils";
import { shapeText } from "../shapes/tensor";
import { extractSequentialModels } from "./sequential";

let parser: Parser;
beforeAll(async () => {
  parser = await loadPythonParserForNode();
});

/** A minimal notebook made of code cells, with optional printed output per cell. */
function notebookOf(...cells: Array<string | { source: string; output: string }>): string {
  return JSON.stringify({
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {},
    cells: cells.map((cell) => ({
      cell_type: "code",
      metadata: {},
      execution_count: 1,
      source: typeof cell === "string" ? cell : cell.source,
      outputs:
        typeof cell === "string"
          ? []
          : [{ output_type: "stream", name: "stdout", text: cell.output }],
    })),
  });
}

function run(text: string) {
  const notebook = parseNotebook(text);
  const analysis = analyzeNotebook(notebook, parser);
  const result = extractSequentialModels(notebook, analysis, parser);
  const summaries = analysis.cells.flatMap((cell) => cell.summaries);
  let ir: RuntimeIR | null = null;
  if (result.wire) {
    const built = buildRuntimeIR(result.wire, { summaries, provenance: "heuristic" });
    if (!built.ok) throw new Error(built.errors.join("\n"));
    ir = built.ir;
  }
  return { result, ir };
}

describe("Keras Sequential", () => {
  it("reads layers, resolves constants and function arguments, and keeps line numbers", () => {
    const { result, ir } = run(
      notebookOf(
        "IMG = 32\nCLASSES = 10",
        [
          "def build_tiny(input_shape, num_classes, width=16):",
          "    model = keras.Sequential([",
          "        # first block",
          "        layers.Conv2D(width, 3, padding='same', activation='relu', input_shape=input_shape),",
          "        layers.MaxPooling2D(),",
          "        layers.Flatten(),",
          "        layers.Dense(num_classes, activation='softmax'),",
          "    ])",
          "    return model",
          "",
          "model = build_tiny((IMG, IMG, 3), num_classes=CLASSES)",
        ].join("\n"),
      ),
    );
    expect(result.declined).toEqual([]);
    const model = ir!.models[0]!;
    expect(model.name).toBe("tiny");
    const ids = ir!.index.topoOrder[model.id]!;
    expect(ids.map((id) => ir!.nodes[id]!.op)).toEqual([
      "input",
      "conv2d",
      "maxpool2d",
      "flatten",
      "dense",
    ]);
    expect(shapeText(ir!.nodes[ids[4]!]!.outShape!)).toBe("(B, 10)");
    // 3·3·3·16+16 = 448, and 16·16·16 · 10 + 10 = 40,970
    expect(ir!.verification.perModel[model.id]!.totals.computed.total).toBe(448 + 40_970);
    expect(ir!.nodes[ids[1]!]!.refs).toEqual([
      { cell: 1, startLine: 4, endLine: 4, role: "declare" },
    ]);
    expect(ir!.groups[model.rootGroupId]!.refs.map((r) => r.role)).toEqual(["definition", "call"]);
    expect(ir!.nodes[ids[1]!]!.provenance).toBe("heuristic");
  });

  it("takes the class count from the notebook's outputs when the code cannot say", () => {
    const { ir } = run(
      notebookOf(
        {
          source: "num_classes = len(class_names)",
          output: "Found 633 images belonging to 8 classes.\n",
        },
        "model = Sequential([Input(shape=(20,)), Dense(num_classes)], name='Head')",
      ),
    );
    expect(ir!.models[0]!.name).toBe("Head");
    expect(ir!.verification.perModel.m_head!.totals.computed.total).toBe(20 * 8 + 8);
    expect(ir!.assumptions.map((entry) => entry.symbol)).toContain("num_classes");
  });

  it("declines instead of guessing", () => {
    const { result } = run(
      notebookOf(
        "augment = Sequential([RandomFlip('horizontal'), RandomRotation(0.1)])",
        "a = Sequential([Dense(4)])",
        "b = Sequential([Input(shape=(8,)), Dense(units)])",
        "c = Sequential(layers_list)",
      ),
    );
    expect(result.wire).toBeNull();
    expect(result.declined.map((entry) => entry.reason)).toEqual([
      'layer "RandomFlip" is not supported yet',
      "the input shape is not stated in the code",
      '"units" of Dense could not be resolved from the code',
      "the layer list contains something that is not a layer call",
    ]);
  });
});

describe("PyTorch nn.Sequential", () => {
  it("reads a classifier and uses the first layer for the input size", () => {
    const { result, ir } = run(
      notebookOf(
        [
          "import torch.nn as nn",
          "net = nn.Sequential(",
          "    nn.Conv2d(3, 8, kernel_size=3, padding=1),",
          "    nn.BatchNorm2d(8),",
          "    nn.ReLU(),",
          "    nn.MaxPool2d(2),",
          "    nn.AdaptiveAvgPool2d(1),",
          "    nn.Flatten(),",
          "    nn.Linear(8, 2),",
          ")",
        ].join("\n"),
      ),
    );
    expect(result.declined).toEqual([]);
    const ids = ir!.index.topoOrder.m_net!;
    expect(shapeText(ir!.nodes[ids[1]!]!.outShape!)).toBe("(B, 8, 224, 224)");
    expect(shapeText(ir!.nodes[ids[4]!]!.outShape!)).toBe("(B, 8, 112, 112)");
    expect(shapeText(ir!.nodes[ids[7]!]!.outShape!)).toBe("(B, 2)");
    expect(ir!.verification.perModel.m_net!.totals.computed).toEqual({
      total: 224 + 16 + 18,
      trainable: 258,
      nonTrainable: 0,
    });
    // The image size is not in the code, so it is listed as an editable assumption.
    expect(ir!.warnings[0]).toContain("224×224 is assumed");
    expect(ir!.assumptions.map((entry) => entry.symbol)).toEqual(["net_image_size"]);
  });

  it("ignores Sequential blocks inside nn.Module classes", () => {
    const { result } = run(
      notebookOf(
        [
          "class Head(nn.Module):",
          "    def __init__(self):",
          "        super().__init__()",
          "        self.fc = nn.Sequential(nn.Linear(8, 4), nn.ReLU())",
        ].join("\n"),
      ),
    );
    expect(result).toEqual({ wire: null, declined: [] });
  });
});

describe("reference notebooks", () => {
  const file = (name: string) => path.join(process.cwd(), "fixtures", "notebooks", `${name}.ipynb`);

  it.skipIf(!existsSync(file("n1-alexnet-vgg")))(
    "N1: AlexNet and VGG16 parse without AI and match the saved summaries exactly",
    () => {
      const { result, ir } = run(readFileSync(file("n1-alexnet-vgg"), "utf8"));
      expect(result.declined).toEqual([]);
      expect(ir!.models.map((model) => model.name)).toEqual(["AlexNet", "VGG16"]);

      const alexnet = ir!.verification.perModel.m_alexnet!;
      expect(alexnet.totals.computed).toEqual({
        total: 46_781_192,
        trainable: 46_780_488,
        nonTrainable: 704,
      });
      expect(alexnet.totals.match).toBe(true);
      expect(alexnet.rows).toHaveLength(16);
      expect(alexnet.rows.every((row) => row.status === "match")).toBe(true);

      const vgg = ir!.verification.perModel.m_vgg16!;
      expect(vgg.totals.computed.total).toBe(14_981_448);
      expect(vgg.totals.match).toBe(true);
      expect(vgg.rows.every((row) => row.status === "match")).toBe(true);

      expect(Object.values(ir!.nodes).filter((node) => node.status !== "ok")).toEqual([]);
      // Conv 1 of AlexNet is on line 12 of cell 14.
      expect(ir!.nodes.n_alexnet_1!.refs).toEqual([
        { cell: 14, startLine: 12, endLine: 12, role: "declare" },
      ]);
    },
  );

  for (const name of ["n2-efficientnet", "n3-mobilenetv2", "n4-vit", "n5-rcnn", "n6-firenet"]) {
    it.skipIf(!existsSync(file(name)))(`${name}: produces no model it is not sure about`, () => {
      const { result, ir } = run(readFileSync(file(name), "utf8"));
      if (ir) {
        for (const report of Object.values(ir.verification.perModel))
          expect(report.totals.match).not.toBe(false);
      }
      expect(Array.isArray(result.declined)).toBe(true);
    });
  }
});
