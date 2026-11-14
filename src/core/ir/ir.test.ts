import { describe, expect, it } from "vitest";
import { CATALOG_OPS, getOp, MVP_OPS, OP_KINDS, opForLayerName } from "../ops";
import type { ModelSummary } from "../notebook/summary";
import { shapeText } from "../shapes/tensor";
import { buildRuntimeIR } from "./build";
import { WireBuilder } from "./builder";
import { alexnetIR } from "./examples";
import { applyRepairPatch } from "./patch";
import type { RuntimeIR } from "./runtime";
import type { WireIR } from "./wire";

function build(wire: unknown, summaries: ModelSummary[] = []): RuntimeIR {
  const result = buildRuntimeIR(wire, { summaries });
  if (!result.ok) throw new Error(result.errors.join("\n"));
  return result.ir;
}

const torch = () =>
  new WireBuilder({ title: "t", summary: "", framework: "pytorch" })
    .model("m", "Net")
    .node("n_in", "input", { shape: [3, 224, 224] });

describe("op catalog", () => {
  it("has a complete entry for every MVP op", () => {
    for (const op of MVP_OPS) {
      const spec = getOp(op);
      expect(spec, op).toBeDefined();
      expect(spec!.title.length, op).toBeGreaterThan(0);
      expect(typeof spec!.codegen.keras, op).toBe("function");
      expect(typeof spec!.explain, op).toBe("function");
    }
    expect(CATALOG_OPS.every((op) => OP_KINDS.includes(op))).toBe(true);
    expect(new Set(CATALOG_OPS).size).toBe(CATALOG_OPS.length);
  });

  it("finds ops by the layer name used in code", () => {
    expect(opForLayerName("tf.keras.layers.Conv2D")).toBe("conv2d");
    expect(opForLayerName("nn.Linear", "pytorch")).toBe("dense");
    expect(opForLayerName("MaxPool2D", "keras")).toBe("maxpool2d");
    expect(opForLayerName("RandomFlip")).toBeUndefined();
  });

  it("writes an explanation and code for every node of AlexNet", () => {
    const ir = build(alexnetIR());
    for (const node of Object.values(ir.nodes)) {
      const spec = getOp(node.op)!;
      const io = { in: node.inShapes, out: node.outShape! };
      expect(spec.explain(node.params, io).length, node.id).toBeGreaterThan(20);
      expect(spec.codegen.keras(node.params, io), node.id).toMatch(/^(layers|keras)\./);
      expect(spec.codegen.pytorch(node.params, io).length, node.id).toBeGreaterThan(5);
    }
    const conv1 = ir.nodes.n_conv1!;
    const io = { in: conv1.inShapes, out: conv1.outShape! };
    expect(getOp("conv2d")!.codegen.keras(conv1.params, io)).toBe(
      'layers.Conv2D(96, (11, 11), strides=(4, 4), padding="valid", activation="relu")',
    );
    expect(getOp("conv2d")!.codegen.pytorch(conv1.params, io)).toBe(
      "nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4), padding=0)",
    );
  });
});

describe("PyTorch rules", () => {
  it("uses NCHW and floor((in + 2p − k) / s) + 1", () => {
    const wire = torch()
      .node(
        "n_conv",
        "conv2d",
        { filters: 64, kernel: 7, stride: 2, padding: 3, bias: false },
        { from: ["n_in"] },
      )
      .node("n_bn", "batchnorm", {}, { from: ["n_conv"] })
      .node("n_pool", "maxpool2d", { pool: 3, stride: 2, padding: 1 }, { from: ["n_bn"] })
      .node("n_gap", "global_avg_pool", {}, { from: ["n_pool"] })
      .node("n_flat", "flatten", {}, { from: ["n_gap"] })
      .node("n_fc", "dense", { units: 10 }, { from: ["n_flat"] })
      .build();
    const ir = build(wire);
    expect(shapeText(ir.nodes.n_conv!.outShape!)).toBe("(B, 64, 112, 112)");
    expect(ir.nodes.n_conv!.paramCount.trainable).toBe(7 * 7 * 3 * 64);
    // PyTorch BatchNorm: running statistics are buffers, not parameters.
    expect(ir.nodes.n_bn!.paramCount).toEqual({ trainable: 128, nonTrainable: 0 });
    expect(shapeText(ir.nodes.n_pool!.outShape!)).toBe("(B, 64, 56, 56)");
    expect(shapeText(ir.nodes.n_gap!.outShape!)).toBe("(B, 64, 1, 1)");
    expect(shapeText(ir.nodes.n_fc!.outShape!)).toBe("(B, 10)");
    expect(ir.nodes.n_fc!.paramCount.trainable).toBe(650);
  });

  it("counts nn.MultiheadAttention as 4E² + 4E", () => {
    const wire = new WireBuilder({ title: "t", summary: "", framework: "pytorch" })
      .model("m", "Net")
      .node("n_in", "input", { shape: [16, 64] })
      .node("n_mha", "multi_head_attention", { numHeads: 4 }, { from: ["n_in"] })
      .build();
    expect(build(wire).nodes.n_mha!.paramCount.trainable).toBe(4 * 64 * 64 + 4 * 64);
  });
});

describe("other ops", () => {
  it("handles depthwise conv, embedding, concat, multiply and slice", () => {
    const wire = new WireBuilder({ title: "t", summary: "", framework: "keras" })
      .model("m", "Net")
      .node("n_in", "input", { shape: [32, 32, 16] })
      .node("n_dw", "depthwise_conv2d", { kernel: 3, padding: "same" }, { from: ["n_in"] })
      .node("n_cat", "concat", {}, { from: ["n_in", "n_dw"] })
      .node("n_gap", "global_avg_pool", { keepDims: true }, { from: ["n_cat"] })
      .node("n_gate", "sigmoid", {}, { from: ["n_gap"] })
      .node("n_scaled", "multiply", {}, { from: ["n_cat", "n_gate"] })
      .node("n_ids", "input", { shape: [12] })
      .node("n_embed", "embedding", { vocab: 1000, dim: 8 }, { from: ["n_ids"] })
      .node("n_first", "slice", { axis: 1, index: 0 }, { from: ["n_embed"] })
      .build();
    const ir = build(wire);
    expect(ir.nodes.n_dw!.paramCount.trainable).toBe(3 * 3 * 16 + 16);
    expect(shapeText(ir.nodes.n_cat!.outShape!)).toBe("(B, 32, 32, 32)");
    expect(shapeText(ir.nodes.n_scaled!.outShape!)).toBe("(B, 32, 32, 32)");
    expect(shapeText(ir.nodes.n_embed!.outShape!)).toBe("(B, 12, 8)");
    expect(ir.nodes.n_embed!.paramCount.trainable).toBe(8000);
    expect(shapeText(ir.nodes.n_first!.outShape!)).toBe("(B, 8)");
  });

  it("moves a frozen layer's weights to non-trainable", () => {
    const wire = new WireBuilder({ title: "t", summary: "", framework: "keras" })
      .model("m", "Net")
      .node("n_in", "input", { shape: [10] })
      .node("n_fc", "dense", { units: 5 }, { from: ["n_in"], trainable: "no" })
      .build();
    expect(build(wire).nodes.n_fc!.paramCount).toEqual({ trainable: 0, nonTrainable: 55 });
  });

  it("keeps declared layers unverified", () => {
    const wire = new WireBuilder({ title: "t", summary: "", framework: "keras" })
      .model("m", "Net")
      .node("n_in", "input", { shape: [224, 224, 3] })
      .node(
        "n_base",
        "backbone",
        { outputShape: [7, 7, 1280], paramCount: 100, nonTrainableCount: 4_049_471 },
        { from: ["n_in"] },
      )
      .node("n_gap", "global_avg_pool", {}, { from: ["n_base"] })
      .build();
    const ir = build(wire);
    expect(ir.nodes.n_base!.status).toBe("unverified");
    expect(shapeText(ir.nodes.n_gap!.outShape!)).toBe("(B, 1280)");
    expect(ir.groups.g_m!.stats.params).toEqual({ trainable: 100, nonTrainable: 4_049_471 });
  });
});

describe("problems are reported, not hidden", () => {
  it("rejects structural garbage", () => {
    const result = buildRuntimeIR({ irVersion: "1.0", nodes: "nope" });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.errors.length).toBeGreaterThan(0);
  });

  it("flags a shape error and marks everything after it unverified", () => {
    const wire = new WireBuilder({ title: "t", summary: "", framework: "keras" })
      .model("m", "Net")
      .node("n_in", "input", { shape: [8, 8, 3] })
      .node(
        "n_conv",
        "conv2d",
        { filters: 4, kernel: 11 },
        { from: ["n_in"], claimed: ["B", 1, 1, 4] },
      )
      .node("n_flat", "flatten", {}, { from: ["n_conv"] })
      .node("n_fc", "dense", { units: 2 }, { from: ["n_flat"] })
      .build();
    const ir = build(wire);
    expect(ir.nodes.n_conv!.status).toBe("error");
    expect(ir.nodes.n_conv!.issues[0]!.message).toContain("does not fit");
    // The claimed shape keeps the rest of the model visible, but untrusted.
    expect(shapeText(ir.nodes.n_fc!.outShape!)).toBe("(B, 2)");
    expect(ir.nodes.n_fc!.status).toBe("unverified");
  });

  it("flags bad params, unknown ops, wrong input counts and dangling edges", () => {
    const wire = new WireBuilder({ title: "t", summary: "", framework: "keras" })
      .model("m", "Net")
      .node("n_in", "input", { shape: [8] })
      .node("n_bad", "dense", { units: "many" }, { from: ["n_in"] })
      .node("n_lstm", "lstm", { units: 4 }, { from: ["n_in"] })
      .node("n_add", "add", {}, { from: ["n_in"] })
      .node("n_orphan", "relu", {}, { from: ["n_missing"] })
      .build();
    const ir = build(wire);
    expect(ir.nodes.n_bad!.status).toBe("error");
    expect(ir.nodes.n_bad!.issues[0]!.code).toBe("bad-param");
    expect(ir.nodes.n_lstm!.status).toBe("unverified");
    expect(ir.nodes.n_lstm!.issues[0]!.code).toBe("op-not-in-catalog");
    expect(ir.nodes.n_add!.issues.some((issue) => issue.code === "wrong-input-count")).toBe(true);
    expect(ir.issues.some((issue) => issue.code === "dangling-edge")).toBe(true);
    expect(ir.nodes.n_orphan!.status).toBe("error");
  });

  it("detects loops", () => {
    const wire = new WireBuilder({ title: "t", summary: "", framework: "keras" })
      .model("m", "Net")
      .node("n_in", "input", { shape: [8] })
      .node("n_a", "add", {}, { from: ["n_in", "n_b"] })
      .node("n_b", "relu", {}, { from: ["n_a"] })
      .build();
    const ir = build(wire);
    expect(ir.issues.some((issue) => issue.code === "cycle")).toBe(true);
    expect(ir.nodes.n_a!.status).toBe("error");
  });

  it("lets the engine win over a wrong claimed shape", () => {
    const wire = alexnetIR();
    wire.nodes.find((node) => node.id === "n_conv1")!.claimedOutputShape = ["B", 55, 55, 96];
    const ir = build(wire);
    expect(shapeText(ir.nodes.n_conv1!.outShape!)).toBe("(B, 54, 54, 96)");
    expect(ir.verification.llmDisagreements).toEqual([
      { nodeId: "n_conv1", claimed: "(B, 55, 55, 96)", computed: "(B, 54, 54, 96)" },
    ]);
  });

  it("marks a node that disagrees with the saved summary", () => {
    const summary: ModelSummary = {
      format: "keras3",
      modelName: "Net",
      rows: [
        {
          name: "dense",
          type: "Dense",
          truncated: false,
          outputShape: [null, 5],
          shapeText: "(None, 5)",
          params: 999,
          connectedTo: [],
          depth: 0,
        },
      ],
      totals: { total: 999, trainable: 999, nonTrainable: 0 },
    };
    const wire = new WireBuilder({ title: "t", summary: "", framework: "keras" })
      .model("m", "Net")
      .node("n_in", "input", { shape: [10] })
      .node("n_fc", "dense", { units: 5 }, { from: ["n_in"] })
      .build();
    const ir = build(wire, [summary]);
    expect(ir.nodes.n_fc!.status).toBe("mismatch");
    expect(ir.verification.perModel.m!.totals.match).toBe(false);
    expect(ir.verification.perModel.m!.rows[0]!.status).toBe("mismatch");
  });
});

describe("applyRepairPatch", () => {
  const empty = { node: null, edge: null, group: null, symbol: null, reason: "test" };

  it("fixes a wrong layer and a symbol without touching the original", () => {
    const wire: WireIR = alexnetIR();
    const wrong = wire.nodes.find((node) => node.id === "n_conv1")!;
    wrong.params = wrong.params.map((entry) =>
      entry.key === "filters" ? { key: "filters", value: 64 } : entry,
    );
    expect(build(wire).verification.perModel.m_alexnet!.totals.computed.total).not.toBe(46_781_192);

    const fixed = {
      ...wrong,
      params: wrong.params.map((entry) =>
        entry.key === "filters" ? { key: "filters", value: 96 } : entry,
      ),
    };
    const result = applyRepairPatch(wire, {
      ops: [
        { ...empty, action: "update_node", targetId: "n_conv1", node: fixed },
        {
          ...empty,
          action: "set_symbol",
          targetId: "num_classes",
          symbol: { name: "num_classes", value: "8" },
        },
        { ...empty, action: "remove_edge", targetId: "e_missing" },
      ],
    });
    expect(result.applied).toBe(2);
    expect(result.errors).toHaveLength(1);
    expect(build(result.ir).verification.perModel.m_alexnet!.totals.computed.total).toBe(
      46_781_192,
    );
    expect(
      wire.nodes
        .find((node) => node.id === "n_conv1")!
        .params.find((entry) => entry.key === "filters")!.value,
    ).toBe(64);
  });

  it("removes a node together with its edges", () => {
    const wire = alexnetIR();
    const result = applyRepairPatch(wire, {
      ops: [{ ...empty, action: "remove_node", targetId: "n_drop2" }],
    });
    expect(result.ir.nodes.some((node) => node.id === "n_drop2")).toBe(false);
    expect(result.ir.edges.some((edge) => edge.from === "n_drop2" || edge.to === "n_drop2")).toBe(
      false,
    );
  });
});
