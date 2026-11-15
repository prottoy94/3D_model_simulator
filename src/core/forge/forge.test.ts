import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { draftToWire, type Draft } from "../ai/draft";
import { exportKeras, exportNotebook, exportPytorch } from "../codegen/export";
import { buildRuntimeIR } from "../ir/build";
import type { RuntimeIR } from "../ir/runtime";
import { simulate } from "../sim/run";
import {
  addNode,
  connect,
  disconnect,
  lastNode,
  PALETTE,
  removeNode,
  setFramework,
  setParam,
  suggestFix,
  TEMPLATES,
} from "./index";

function build(doc: Draft): RuntimeIR {
  const result = buildRuntimeIR(draftToWire(doc), { provenance: "user" });
  if (!result.ok) throw new Error(result.errors.join("\n"));
  return result.ir;
}
const template = (id: string) => TEMPLATES.find((entry) => entry.id === id)!.build();
const problems = (ir: RuntimeIR) =>
  Object.values(ir.nodes).filter((node) => node.status === "error");
const total = (ir: RuntimeIR) => ir.verification.perModel.m_1!.totals.computed.total;

describe("templates", () => {
  for (const entry of TEMPLATES) {
    it(`${entry.label} is valid in Keras and in PyTorch, with the same number of weights`, () => {
      const keras = build(entry.build());
      const torch = build(setFramework(entry.build(), "pytorch"));
      expect(problems(keras)).toEqual([]);
      expect(problems(torch)).toEqual([]);
      // PyTorch does not count BatchNorm's running statistics as parameters.
      expect(keras.verification.perModel.m_1!.totals.computed.trainable).toBe(total(torch));
    });
  }

  it("every palette layer has defaults the catalog accepts", () => {
    for (const entry of PALETTE) {
      const { doc, id } = addNode(template("small-cnn"), entry.op, "conv2d_1");
      const node = build(doc).nodes[id]!;
      expect(
        node.issues.filter((issue) => issue.code === "bad-param"),
        entry.op,
      ).toEqual([]);
    }
  });
});

describe("editing", () => {
  it("inserts a layer in the middle of a chain and appends at the end", () => {
    const start = template("small-cnn");
    const { doc, id } = addNode(start, "batchnorm", "conv2d_1");
    const nodes = doc.models[0]!.nodes;
    expect(nodes.find((node) => node.id === id)!.inputs).toEqual(["conv2d_1"]);
    expect(nodes.find((node) => node.id === "maxpool2d_1")!.inputs).toEqual([id]);
    expect(start.models[0]!.nodes).toHaveLength(8); // the original is untouched

    const appended = addNode(doc, "dense", null);
    expect(appended.doc.models[0]!.nodes.find((node) => node.id === appended.id)!.inputs).toEqual([
      "dense_1",
    ]);
    expect(lastNode(appended.doc)).toBe(appended.id);
    expect(problems(build(appended.doc))).toEqual([]);
  });

  it("removes a layer and reconnects its neighbours", () => {
    const doc = removeNode(template("small-cnn"), "maxpool2d_1");
    expect(doc.models[0]!.nodes.find((node) => node.id === "conv2d_2")!.inputs).toEqual([
      "conv2d_1",
    ]);
    expect(problems(build(doc))).toEqual([]);
  });

  it("changes a setting, the label and the parameter count", () => {
    const before = total(build(template("small-cnn")));
    const doc = setParam(template("small-cnn"), "conv2d_2", "filters", 128);
    expect(doc.models[0]!.nodes.find((node) => node.id === "conv2d_2")!.label).toBe(
      "Conv 128@3×3 · relu",
    );
    expect(total(build(doc))).toBeGreaterThan(before);
  });

  it("connects a skip connection and refuses loops", () => {
    let doc = addNode(template("small-cnn"), "add", "conv2d_2").doc;
    doc = connect(doc, "maxpool2d_1", "add_1");
    expect(doc.models[0]!.nodes.find((node) => node.id === "add_1")!.inputs).toEqual([
      "conv2d_2",
      "maxpool2d_1",
    ]);
    expect(connect(doc, "dense_1", "conv2d_1")).toBe(doc); // would be a loop
    expect(connect(doc, "conv2d_1", "input_1")).toBe(doc); // inputs take nothing
    expect(
      disconnect(doc, "maxpool2d_1", "add_1").models[0]!.nodes.find((node) => node.id === "add_1")!
        .inputs,
    ).toEqual(["conv2d_2"]);
  });

  it("swaps the image layout when the framework changes, and back", () => {
    const torch = setFramework(template("small-cnn"), "pytorch");
    expect(torch.models[0]!.nodes[0]!.params.shape).toEqual([3, 64, 64]);
    expect(setFramework(torch, "keras").models[0]!.nodes[0]!.params.shape).toEqual([64, 64, 3]);
  });
});

describe("shape errors and suggestions", () => {
  it("catches a kernel larger than the image and says how to fix it", () => {
    let doc = template("small-cnn");
    for (const id of ["conv2d_1", "conv2d_2"]) doc = setParam(doc, id, "padding", "valid");
    doc = setParam(doc, "conv2d_2", "kernel", [40, 40]);
    const ir = build(doc);
    const broken = ir.nodes.conv2d_2!;
    expect(broken.status).toBe("error");
    expect(suggestFix(broken)).toContain('Set padding to "same"');
    expect(suggestFix(ir.nodes.conv2d_1!)).toBeNull();
  });

  it("explains a merge with one input, mismatched branches and a misplaced conv", () => {
    const lonely = build(addNode(template("small-cnn"), "add", "conv2d_2").doc);
    expect(suggestFix(lonely.nodes.add_1!)).toContain("Connect a second layer");

    let doc = addNode(template("small-cnn"), "add", "conv2d_2").doc;
    doc = connect(doc, "conv2d_1", "add_1"); // 64×64×32 + 32×32×64
    expect(suggestFix(build(doc).nodes.add_1!)).toContain("same shape");

    const misplaced = build(addNode(template("small-cnn"), "conv2d", "global_avg_pool_1").doc);
    expect(suggestFix(misplaced.nodes.conv2d_3!)).toContain("needs an image");
  });
});

describe("export", () => {
  const cnn = template("small-cnn");

  it("writes a Keras functional script with one line per layer", () => {
    const code = exportKeras(build(cnn), "m_1");
    expect(code.warnings).toEqual([]);
    expect(code.text).toContain("def build_smallcnn():");
    expect(code.text).toContain("    input_1 = keras.Input(shape=(64, 64, 3))  # (B, 64, 64, 3)");
    expect(code.text).toContain(
      '    conv2d_1 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="relu")(input_1)',
    );
    expect(code.text).toContain('    return keras.Model(input_1, dense_1, name="SmallCNN")');
    const lines = code.text.split("\n");
    expect(lines[code.lineOfNode.maxpool2d_2! - 1]).toContain("maxpool2d_2 = layers.MaxPooling2D");
  });

  it("writes a PyTorch module with fused activations and a residual add", () => {
    const ir = build(setFramework(template("residual"), "pytorch"));
    const code = exportPytorch(ir, "m_1");
    expect(code.warnings).toEqual([]);
    expect(code.text).toContain("class ResidualNet(nn.Module):");
    expect(code.text).toContain(
      '        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding="same")',
    );
    expect(code.text).toContain("        self.batchnorm_1 = nn.BatchNorm2d(32)");
    expect(code.text).toContain(
      "        conv2d_1 = F.relu(self.conv2d_1(input_1))  # (B, 32, 64, 64)",
    );
    expect(code.text).toContain("        add_1 = conv2d_3 + conv2d_1");
    expect(code.text).toContain("        global_avg_pool_1 = relu_1.mean(dim=(2, 3))");
    expect(code.text).toContain(
      "        dense_1 = F.softmax(self.dense_1(global_avg_pool_1), dim=-1)",
    );
    expect(code.text).toContain("out = model(torch.randn(1, 3, 64, 64))");
  });

  it("writes attention with PyTorch's (query, key, value) order", () => {
    const code = exportPytorch(build(setFramework(template("transformer"), "pytorch")), "m_1");
    expect(code.text).toContain(
      "self.attention_1 = nn.MultiheadAttention(64, 4, batch_first=True)",
    );
    expect(code.text).toContain(
      "attention_1 = self.attention_1(layernorm_1, layernorm_1, layernorm_1)[0]",
    );
    expect(code.text).toContain("global_avg_pool_1 = add_2.mean(dim=1)");
  });

  it("wraps the script as a notebook", () => {
    const notebook = JSON.parse(exportNotebook(exportKeras(build(cnn), "m_1"), "SmallCNN")) as {
      cells: Array<{ cell_type: string; source: string }>;
    };
    expect(notebook.cells.map((cell) => cell.cell_type)).toEqual(["markdown", "code"]);
    expect(notebook.cells[1]!.source).toContain("model.summary()");
  });

  it("writes every template to disk for a Python syntax check", () => {
    // `python -m py_compile` on these files is run outside the test suite (see docs/08).
    const dir = path.join(process.cwd(), ".cache", "forge-export");
    mkdirSync(dir, { recursive: true });
    for (const entry of TEMPLATES) {
      const keras = build(entry.build());
      const torch = build(setFramework(entry.build(), "pytorch"));
      writeFileSync(path.join(dir, `${entry.id}_keras.py`), exportKeras(keras, "m_1").text);
      writeFileSync(path.join(dir, `${entry.id}_torch.py`), exportPytorch(torch, "m_1").text);
    }
    expect(TEMPLATES.length).toBe(5);
  });
});

describe("a Forge model can be simulated", () => {
  it("runs the residual template end to end with real values", () => {
    const ir = build(template("residual"));
    const results = [...simulate(ir, "m_1", { kind: "shapes" })];
    expect(results).toHaveLength(9);
    expect(results.every((result) => !result.tensor.illustrative)).toBe(true);
    const output = results[results.length - 1]!.tensor.data;
    expect([...output].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
  });
});
