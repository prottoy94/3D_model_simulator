import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { Parser } from "web-tree-sitter";
import { buildRuntimeIR } from "../ir/build";
import type { RuntimeIR } from "../ir/runtime";
import { analyzeNotebook } from "../notebook/analyze-notebook";
import { parseNotebook } from "../notebook/parse";
import { parseSummaries } from "../notebook/summary";
import { CATALOG_OPS } from "../ops";
import { loadPythonParserForNode } from "../python/test-utils";
import { shapeText } from "../shapes/tensor";
import { resolveZoo } from "../zoo";
import { buildExtractionContext, describeProblems } from "./context";
import {
  applyDraftPatch,
  draftToWire,
  evaluateExpression,
  extractJson,
  parseDraft,
  parseDraftPatch,
  type Draft,
} from "./draft";
import { OP_REFERENCE, systemPrompt } from "./prompt";

const summaries = (file: string) =>
  parseSummaries(
    readFileSync(path.join(process.cwd(), "fixtures", "summaries", `${file}.txt`), "utf8"),
  );

function build(draft: Draft, summaryFile?: string): RuntimeIR {
  const result = buildRuntimeIR(draftToWire(draft), {
    summaries: summaryFile ? summaries(summaryFile) : [],
  });
  if (!result.ok) throw new Error(result.errors.join("\n"));
  return result.ir;
}

function draftOf(json: unknown): Draft {
  const parsed = parseDraft(JSON.stringify(json));
  if (!parsed.ok) throw new Error(parsed.errors.join("\n"));
  return parsed.value;
}

/** What a model should write for the EfficientNet and MobileNet notebooks (N2, N3). */
function transferDraft(zoo: string): Draft {
  const draft = draftOf({
    title: "Transfer learning",
    framework: "keras",
    symbols: [{ name: "num_classes", value: 8, source: "from-output" }],
    models: [
      {
        id: "m",
        name: "classifier",
        nodes: [
          { id: "in", op: "input", params: { shape: [224, 224, 3] } },
          {
            id: "base",
            op: "backbone",
            inputs: ["in"],
            params: { zoo, inputSize: [224, 224] },
            frozen: true,
          },
          { id: "gap", op: "GlobalAveragePooling2D", inputs: ["base"] },
          { id: "fc", op: "dense", inputs: ["gap"], params: { units: 128, activation: "relu" } },
          { id: "drop", op: "dropout", inputs: ["fc"], params: { rate: 0.3 } },
          {
            id: "out",
            op: "Dense",
            inputs: ["drop"],
            params: { units: "num_classes", activation: "softmax" },
          },
        ],
      },
    ],
  });
  // Every node points at a line of code, as the prompt requires.
  draft.models[0]!.nodes.forEach((node, index) => {
    node.lines = [16, index + 1, index + 1];
  });
  return draft;
}

describe("prompt", () => {
  it("documents every op in the catalog", () => {
    for (const op of CATALOG_OPS) expect(OP_REFERENCE[op], op).toBeTruthy();
    const prompt = systemPrompt();
    expect(prompt).toContain("- conv2d:");
    expect(prompt).toContain("EfficientNetB0");
    expect(prompt.length).toBeLessThan(9000);
  });
});

describe("draft parsing", () => {
  it("accepts JSON wrapped in prose or a code fence", () => {
    expect(extractJson('Here you go:\n```json\n{"a": 1}\n```')).toEqual({ a: 1 });
    expect(parseDraft("no json here").ok).toBe(false);
  });

  it("reports what is missing in plain words", () => {
    const result = parseDraft('{"models": [{"id": "m", "nodes": []}]}');
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.join(" ")).toContain("models.0.name");
  });
});

describe("draft → verified IR", () => {
  it("N2: EfficientNetB0 + head matches the notebook's totals exactly", () => {
    const ir = build(transferDraft("EfficientNetB0"), "n2-efficientnet-cell19");
    const report = ir.verification.perModel.m_1!;
    expect(shapeText(ir.nodes.base!.outShape!)).toBe("(B, 7, 7, 1280)");
    expect(report.totals.computed).toEqual({
      total: 4_214_571,
      trainable: 165_000,
      nonTrainable: 4_049_571,
    });
    expect(report.totals.match).toBe(true);
    // Only the head is compared row by row; the backbone's 230 rows are not "unmatched".
    expect(report.rows.map((row) => row.status)).toEqual(["match", "match", "match", "match"]);
    expect(report.unmatchedRows).toEqual([]);
    expect(describeProblems(ir)).toEqual([]);
    expect(ir.nodes.base!.customOpName).toBe("keras.applications.EfficientNetB0");
  });

  it("N3: MobileNetV2 + head matches the notebook's totals exactly", () => {
    const ir = build(transferDraft("tf.keras.applications.MobileNetV2"), "n3-mobilenetv2-cell18");
    expect(ir.verification.perModel.m_1!.totals.computed).toEqual({
      total: 2_422_984,
      trainable: 165_000,
      nonTrainable: 2_257_984,
    });
    expect(ir.verification.perModel.m_1!.totals.match).toBe(true);
  });

  it("N4: a repeated transformer block written once unrolls to 404,040 params", () => {
    const draft = draftOf({
      framework: "keras",
      symbols: [
        { name: "TRANSFORMER_LAYERS", value: 4 },
        { name: "PROJECTION_DIM", value: 64 },
      ],
      models: [
        {
          id: "vit",
          name: "ViT_scratch",
          groups: [
            { id: "enc", label: "PatchEncoder", kind: "module" },
            {
              id: "blk",
              label: "Transformer block",
              repeat: "TRANSFORMER_LAYERS",
              lines: [16, 68, 69],
            },
          ],
          nodes: [
            { id: "in", op: "input", params: { shape: [224, 224, 3] } },
            {
              id: "patches",
              op: "patchify",
              className: "Patches",
              inputs: ["in"],
              params: { patch: 16 },
            },
            {
              id: "proj",
              op: "patch_embed",
              group: "enc",
              inputs: ["patches"],
              params: { dim: "PROJECTION_DIM" },
            },
            { id: "pos", op: "pos_embed", group: "enc", inputs: ["proj"], params: {} },
            {
              id: "ln1",
              op: "LayerNormalization",
              group: "blk",
              inputs: ["pos"],
              params: { epsilon: 1e-6 },
            },
            {
              id: "att",
              op: "multi_head_attention",
              group: "blk",
              inputs: ["ln1", "ln1"],
              params: { numHeads: 4, keyDim: 64 },
            },
            { id: "add1", op: "add", group: "blk", inputs: ["att", "pos"] },
            { id: "ln2", op: "layernorm", group: "blk", inputs: ["add1"] },
            {
              id: "d1",
              op: "dense",
              group: "blk",
              inputs: ["ln2"],
              params: { units: 128, activation: "gelu" },
            },
            { id: "d2", op: "dense", group: "blk", inputs: ["d1"], params: { units: 64 } },
            { id: "add2", op: "add", group: "blk", inputs: ["d2", "add1"] },
            { id: "ln", op: "layernorm", inputs: ["add2"] },
            { id: "gap", op: "global_avg_pool", inputs: ["ln"] },
            { id: "fc", op: "dense", inputs: ["gap"], params: { units: 128, activation: "gelu" } },
            {
              id: "out",
              op: "dense",
              inputs: ["fc"],
              params: { units: 8, activation: "softmax" },
              lines: [16, 77, 77],
            },
          ],
          outputs: ["out"],
        },
      ],
    });
    const ir = build(draft, "n4-vit-cell18");
    const report = ir.verification.perModel.m_1!;
    expect(report.totals.computed.total).toBe(404_040);
    expect(report.totals.match).toBe(true);
    expect(Object.keys(ir.nodes).filter((id) => id.startsWith("att#"))).toHaveLength(4);
    expect(ir.nodes.out!.refs).toEqual([{ cell: 16, startLine: 77, endLine: 77, role: "declare" }]);
  });

  it("keeps two models apart even when they reuse ids, and maps unknown layers to custom", () => {
    const ir = build(
      draftOf({
        models: [
          {
            id: "a",
            name: "A",
            nodes: [
              { id: "in", op: "input", params: { shape: [8] } },
              { id: "out", op: "dense", inputs: ["in"], params: { units: 2 } },
            ],
          },
          {
            id: "b",
            name: "B",
            nodes: [
              { id: "in", op: "input", params: { shape: [8] } },
              {
                id: "odd",
                op: "MyFancyLayer",
                inputs: ["in"],
                outputShape: ["B", 5],
                params: { paramCount: 7 },
              },
              { id: "out", op: "dense", inputs: ["odd"], params: { units: 3 } },
            ],
          },
        ],
      }),
    );
    expect(ir.models.map((model) => model.name)).toEqual(["A", "B"]);
    expect(ir.verification.perModel.m_1!.totals.computed.total).toBe(18);
    expect(ir.nodes["odd@2"]!.op).toBe("custom");
    expect(ir.nodes["odd@2"]!.customOpName).toBe("MyFancyLayer");
    expect(ir.verification.perModel.m_2!.totals.computed.total).toBe(7 + 18);
  });
});

describe("repair", () => {
  it("describes a wrong head, and a patch fixes it", () => {
    const wrong = transferDraft("EfficientNetB0");
    wrong.models[0]!.nodes.find((node) => node.id === "fc")!.params.units = 256;
    const broken = build(wrong, "n2-efficientnet-cell19");
    const problems = describeProblems(broken);
    expect(problems.some((problem) => problem.includes('node "fc"'))).toBe(true);
    expect(problems.some((problem) => problem.includes("4,214,571"))).toBe(true);

    const patch = parseDraftPatch(
      JSON.stringify({
        nodes: [
          { id: "fc", op: "dense", inputs: ["gap"], params: { units: 128, activation: "relu" } },
        ],
      }),
    );
    if (!patch.ok) throw new Error(patch.errors.join("\n"));
    const fixed = build(applyDraftPatch(wrong, patch.value), "n2-efficientnet-cell19");
    expect(describeProblems(fixed)).toEqual([]);
    expect(wrong.models[0]!.nodes.find((node) => node.id === "fc")!.params.units).toBe(256);
  });

  it("removes nodes together with the references to them", () => {
    const draft = transferDraft("EfficientNetB0");
    const patch = parseDraftPatch(
      '{"remove": ["drop"], "nodes": [{"id": "out", "op": "dense", "inputs": ["fc"], "params": {"units": 8}}]}',
    );
    if (!patch.ok) throw new Error(patch.errors.join("\n"));
    const next = applyDraftPatch(draft, patch.value);
    expect(next.models[0]!.nodes.map((node) => node.id)).toEqual([
      "in",
      "base",
      "gap",
      "fc",
      "out",
    ]);
  });
});

describe("forgiving values", () => {
  it("works out symbol names and arithmetic inside params", () => {
    const symbols = new Map<string, unknown>([
      ["num_classes", 8],
      ["IMG", 224],
    ]);
    expect(evaluateExpression("num_classes + 1", symbols)).toBe(9);
    expect(evaluateExpression("(IMG // 32) ** 2 * 9", symbols)).toBe(441);
    expect(evaluateExpression("os.system('x')", symbols)).toBeUndefined();
    expect(evaluateExpression("unknown + 1", symbols)).toBeUndefined();

    const ir = build(
      draftOf({
        symbols: [
          { name: "IMG", value: 32 },
          { name: "num_classes", value: 8 },
        ],
        models: [
          {
            id: "m",
            name: "M",
            nodes: [
              { id: "in", op: "input", params: { shape: ["IMG", "IMG", 3] } },
              { id: "gap", op: "global_avg_pool", inputs: ["in"] },
              { id: "out", op: "dense", inputs: ["gap"], params: { units: "num_classes + 1" } },
            ],
          },
        ],
      }),
    );
    expect(shapeText(ir.nodes.in!.outShape!)).toBe("(B, 32, 32, 3)");
    expect(shapeText(ir.nodes.out!.outShape!)).toBe("(B, 9)");
  });

  it("reports only the root cause, and asks for missing line numbers", () => {
    const ir = build(
      draftOf({
        models: [
          {
            id: "m",
            name: "M",
            nodes: [
              { id: "in", op: "input", params: { shape: "big" } },
              { id: "conv", op: "conv2d", inputs: ["in"], params: { filters: 4, kernel: 3 } },
              { id: "out", op: "dense", inputs: ["conv"], params: { units: 2 } },
            ],
          },
        ],
      }),
    );
    const problems = describeProblems(ir);
    expect(problems.filter((problem) => problem.startsWith("node"))).toHaveLength(1);
    expect(problems[0]).toContain('node "in"');
    expect(problems.some((problem) => problem.includes('have no "lines"'))).toBe(true);
  });

  it("a patch changes only the fields it gives", () => {
    const draft = transferDraft("EfficientNetB0");
    draft.models[0]!.nodes.find((node) => node.id === "fc")!.lines = [16, 5, 5];
    const patch = parseDraftPatch('{"nodes": [{"id": "fc", "params": {"units": 64}}]}');
    if (!patch.ok) throw new Error(patch.errors.join("; "));
    const fc = applyDraftPatch(draft, patch.value).models[0]!.nodes.find(
      (node) => node.id === "fc",
    )!;
    expect(fc.params).toEqual({ units: 64 });
    expect(fc.lines).toEqual([16, 5, 5]);
    expect(fc.op).toBe("dense");
    expect(fc.inputs).toEqual(["gap"]);
  });
});

describe("zoo facts", () => {
  it("scales the output with the input size and knows token models", () => {
    expect(resolveZoo("EfficientNetB0", [96, 96])!.outputShape).toEqual([3, 3, 1280]);
    expect(resolveZoo("vit_base_patch16_224_imagenet")!.outputShape).toEqual([197, 768]);
    expect(resolveZoo("models.convnext_tiny", [224, 224], "pytorch")!.outputShape).toEqual([
      768, 7, 7,
    ]);
    expect(resolveZoo("SomethingElse")).toBeNull();
  });
});

describe("context slicer", () => {
  let parser: Parser;
  beforeAll(async () => {
    parser = await loadPythonParserForNode();
  });

  it("sends model and setup code with line numbers, and nothing else", () => {
    const text = JSON.stringify({
      nbformat: 4,
      nbformat_minor: 5,
      metadata: {},
      cells: [
        {
          cell_type: "code",
          metadata: {},
          outputs: [],
          source: "import pandas as pd\ndf = pd.read_csv('secret_patients.csv')",
        },
        { cell_type: "code", metadata: {}, outputs: [], source: "IMG = 32" },
        {
          cell_type: "code",
          metadata: {},
          outputs: [],
          source: "model = Sequential([\n    Dense(4, input_shape=(IMG,)),\n])",
        },
        {
          cell_type: "code",
          metadata: {},
          outputs: [],
          source: "plt.plot(history.history['loss'])",
        },
      ],
    });
    const notebook = parseNotebook(text);
    const context = buildExtractionContext(notebook, analyzeNotebook(notebook, parser));
    expect(context.cells).toEqual([2]);
    expect(context.text).toContain("### cell 2 [model]");
    expect(context.text).toContain("   2|     Dense(4, input_shape=(IMG,)),");
    expect(context.text).toContain("IMG = 32");
    expect(context.text).not.toContain("secret_patients");
    expect(context.text).not.toContain("plt.plot");
    expect(context.approxTokens).toBe(Math.ceil(context.chars / 4));
  });

  const n2 = path.join(process.cwd(), "fixtures", "notebooks", "n2-efficientnet.ipynb");
  it.skipIf(!existsSync(n2))("N2: shortens the 240-row backbone summary and stays small", () => {
    const notebook = parseNotebook(readFileSync(n2, "utf8"));
    const context = buildExtractionContext(notebook, analyzeNotebook(notebook, parser));
    expect(context.text).toContain("rows omitted");
    expect(context.text).toContain("total 4,214,571, trainable 165,000");
    expect(context.cells).toEqual(expect.arrayContaining([16, 17]));
    expect(context.approxTokens).toBeLessThan(12_000);
  });
});
