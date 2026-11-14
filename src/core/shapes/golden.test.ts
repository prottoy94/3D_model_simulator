/**
 * Golden facts from docs/07-NOTEBOOK-PATTERNS.md §4. Every number here was
 * printed by the reference notebooks' own `model.summary()`; the engine must
 * reproduce all of them from the IR alone.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { buildRuntimeIR } from "../ir/build";
import { ref, WireBuilder } from "../ir/builder";
import { EXAMPLE_IRS } from "../ir/examples";
import type { RuntimeIR } from "../ir/runtime";
import type { ParamValue } from "../ir/wire";
import { parseSummaries } from "../notebook/summary";
import { shapeText } from "./tensor";

const root = process.cwd();
const fixture = (name: string): unknown =>
  JSON.parse(readFileSync(path.join(root, "fixtures", "ir", `${name}.json`), "utf8"));
const summaries = (file: string) =>
  parseSummaries(readFileSync(path.join(root, "fixtures", "summaries", `${file}.txt`), "utf8"));

function build(wire: unknown, summaryFile?: string): RuntimeIR {
  const result = buildRuntimeIR(wire, { summaries: summaryFile ? summaries(summaryFile) : [] });
  if (!result.ok) throw new Error(result.errors.join("\n"));
  return result.ir;
}

const shape = (ir: RuntimeIR, id: string) => shapeText(ir.nodes[id]!.outShape!);
const params = (ir: RuntimeIR, id: string) => {
  const count = ir.nodes[id]!.paramCount;
  return count.trainable + count.nonTrainable;
};
const problems = (ir: RuntimeIR) =>
  Object.values(ir.nodes)
    .filter((node) => node.status !== "ok")
    .map((node) => `${node.id}: ${node.status}`);

describe("fixtures/ir matches src/core/ir/examples.ts", () => {
  for (const [name, make] of Object.entries(EXAMPLE_IRS)) {
    it(name, () => expect(fixture(name)).toEqual(make()));
  }
});

describe("N1 AlexNet", () => {
  const ir = build(fixture("alexnet"), "n1-alexnet-vgg-cell16");
  const report = ir.verification.perModel.m_alexnet!;

  it("computes the documented shapes and counts", () => {
    expect(shape(ir, "n_conv1")).toBe("(B, 54, 54, 96)");
    expect(params(ir, "n_conv1")).toBe(34_944);
    expect(params(ir, "n_bn1")).toBe(384);
    expect(shape(ir, "n_pool1")).toBe("(B, 26, 26, 96)");
    expect(shape(ir, "n_conv2")).toBe("(B, 26, 26, 256)");
    expect(params(ir, "n_conv2")).toBe(614_656);
    expect(shape(ir, "n_pool2")).toBe("(B, 12, 12, 256)");
    expect(params(ir, "n_conv3")).toBe(885_120);
    expect(params(ir, "n_conv4")).toBe(1_327_488);
    expect(shape(ir, "n_flatten")).toBe("(B, 6400)");
    expect(params(ir, "n_fc1")).toBe(26_218_496);
    expect(shape(ir, "n_fc3")).toBe("(B, 8)");
  });

  it("matches the notebook's totals, including BatchNorm's non-trainable 704", () => {
    expect(report.totals.computed).toEqual({
      total: 46_781_192,
      trainable: 46_780_488,
      nonTrainable: 704,
    });
    expect(report.totals.match).toBe(true);
  });

  it("matches every row of the saved summary", () => {
    expect(report.rows).toHaveLength(16);
    expect(report.rows.filter((row) => row.status !== "match")).toEqual([]);
    expect(report.unmatchedRows).toEqual([]);
    expect(problems(ir)).toEqual([]);
  });

  it("counts FLOPs for conv1: 2 · 54·54 · 11·11·3 · 96", () => {
    expect(ir.nodes.n_conv1!.flops).toBe(2 * 54 * 54 * 11 * 11 * 3 * 96);
  });
});

describe("N1 VGG16 variant", () => {
  const ir = build(fixture("vgg16"), "n1-alexnet-vgg-cell28");
  const report = ir.verification.perModel.m_vgg16!;

  it("matches the totals: 14,981,448, all trainable", () => {
    expect(report.totals.computed).toEqual({
      total: 14_981_448,
      trainable: 14_981_448,
      nonTrainable: 0,
    });
    expect(report.totals.match).toBe(true);
  });

  it("splits into conv stack, Dense 512 and Dense 8", () => {
    const convs = Object.values(ir.nodes).filter((node) => node.op === "conv2d");
    expect(convs).toHaveLength(13);
    expect(convs.reduce((sum, node) => sum + node.paramCount.trainable, 0)).toBe(14_714_688);
    expect(params(ir, "n_fc1")).toBe(262_656);
    expect(params(ir, "n_fc2")).toBe(4_104);
    expect(shape(ir, "n_pool5")).toBe("(B, 7, 7, 512)");
    expect(shape(ir, "n_gap")).toBe("(B, 512)");
  });

  it("matches every row of the saved summary", () => {
    expect(report.rows.filter((row) => row.status !== "match")).toEqual([]);
    expect(report.unmatchedRows).toEqual([]);
    expect(problems(ir)).toEqual([]);
  });

  it("reports stage totals on groups", () => {
    expect(ir.groups.g_block1!.stats.params.trainable).toBe(1_792 + 36_928);
    expect(ir.groups.g_vgg16!.stats.params.trainable).toBe(14_981_448);
  });
});

describe("N4 mini-ViT", () => {
  const ir = build(fixture("mini-vit"), "n4-vit-cell18");
  const report = ir.verification.perModel.m_vit!;

  it("cuts the image into 196 tokens of 768 numbers", () => {
    expect(shape(ir, "n_patches")).toBe("(B, 196, 768)");
    expect(shape(ir, "n_pos")).toBe("(B, 196, 64)");
  });

  it("counts PatchEncoder, LayerNorm and attention like Keras", () => {
    expect(ir.groups.g_patch_encoder!.stats.params.trainable).toBe(61_760);
    expect(params(ir, "n_ln1#1")).toBe(128);
    expect(params(ir, "n_mha#1")).toBe(66_368);
    expect(params(ir, "n_mlp1#3")).toBe(8_320);
    expect(params(ir, "n_mlp2#4")).toBe(8_256);
  });

  it("unrolls the transformer block four times and chains the copies", () => {
    expect(Object.keys(ir.nodes).filter((id) => id.startsWith("n_mha#"))).toHaveLength(4);
    expect(ir.nodes["n_mha#2"]!.templateId).toBe("n_mha");
    expect(ir.groups["g_blocks#2"]!.templateId).toBe("g_blocks");
    // Block 2 reads block 1's output on both the main path and the residual.
    const intoLn = ir.index.inEdges["n_ln1#2"]!.map((edge) => edge.from);
    const intoAdd = ir.index.inEdges["n_add1#2"]!.map((edge) => edge.from);
    expect(intoLn).toEqual(["n_add2#1"]);
    expect(intoAdd).toEqual(["n_mha#2", "n_add2#1"]);
    expect(ir.index.inEdges.n_ln_out!.map((edge) => edge.from)).toEqual(["n_add2#4"]);
  });

  it("matches the total of 404,040 and every summary row", () => {
    expect(report.totals.computed).toEqual({ total: 404_040, trainable: 404_040, nonTrainable: 0 });
    expect(report.totals.match).toBe(true);
    expect(report.rows.filter((row) => row.status !== "match")).toEqual([]);
    expect(report.unmatchedRows).toEqual([]);
    expect(problems(ir)).toEqual([]);
  });
});

// ── N5: the R-CNN family, written with the builder ────────────────────

const relu = (filters: number): Record<string, ParamValue> => ({
  filters,
  kernel: [3, 3],
  padding: "same",
  activation: "relu",
});

/** Three Conv+Pool blocks. Returns the id of the last pooling node. */
function smallBackbone(b: WireBuilder, from: string): string {
  let previous = from;
  [32, 64, 128].forEach((filters, index) => {
    b.node(`n_conv${index + 1}`, "conv2d", relu(filters), { from: [previous] });
    b.node(`n_pool${index + 1}`, "maxpool2d", { pool: [2, 2] }, { from: [`n_conv${index + 1}`] });
    previous = `n_pool${index + 1}`;
  });
  return previous;
}

function rcnnHead(name: string, units: number, activation: string) {
  const b = new WireBuilder({ title: name, summary: "", framework: "keras", task: "detection" })
    .model("m", name)
    .node("n_in", "input", { shape: [64, 64, 3] });
  const features = smallBackbone(b, "n_in");
  return b
    .node("n_flatten", "flatten", {}, { from: [features] })
    .node("n_fc", "dense", { units: 128, activation: "relu" }, { from: ["n_flatten"] })
    .node("n_drop", "dropout", { rate: 0.3 }, { from: ["n_fc"] })
    .node("n_out", "dense", { units, activation }, { from: ["n_drop"] })
    .outputs("n_out")
    .build();
}

describe("N5 R-CNN family", () => {
  it("RCNN_Classifier: 1,143,113 params", () => {
    const ir = build(rcnnHead("RCNN_Classifier", 9, "softmax"), "n5-rcnn-cell26");
    const report = ir.verification.perModel.m!;
    expect(shape(ir, "n_flatten")).toBe("(B, 8192)");
    expect(report.totals.computed.total).toBe(1_143_113);
    expect(report.totals.match).toBe(true);
    expect(report.rows.filter((row) => row.status !== "match")).toEqual([]);
    expect(report.unmatchedRows).toEqual([]);
  });

  it("RCNN_BBox_Regressor: 1,142,468 params (computed)", () => {
    const ir = build(rcnnHead("RCNN_BBox_Regressor", 4, "linear"));
    expect(ir.verification.perModel.m!.totals.computed.total).toBe(1_142_468);
    expect(ir.verification.perModel.m!.totals.match).toBeNull();
  });

  it("FastRCNN: RoI pooling and TimeDistributed layers, 897,869 params", () => {
    const b = new WireBuilder({
      title: "FastRCNN",
      summary: "",
      framework: "keras",
      task: "detection",
    })
      .model("m", "FastRCNN")
      .node("n_image", "input", { shape: [224, 224, 3] })
      .node("n_rois", "input", { shape: [-1, 4], dynamicDim: "R" });
    const features = smallBackbone(b, "n_image");
    const wire = b
      .node(
        "n_roi_pool",
        "roi_pool",
        { pooled: 7 },
        { from: [features, "n_rois"], customOpName: "ROIPoolingLayer" },
      )
      .node("n_flatten", "flatten", { timeDistributed: true }, { from: ["n_roi_pool"] })
      .node(
        "n_fc",
        "dense",
        { units: 128, activation: "relu", timeDistributed: true },
        { from: ["n_flatten"] },
      )
      .node("n_drop", "dropout", { rate: 0.3, timeDistributed: true }, { from: ["n_fc"] })
      .node(
        "n_cls",
        "dense",
        { units: 9, activation: "softmax", timeDistributed: true },
        { from: ["n_drop"] },
      )
      .node("n_box", "dense", { units: 4, timeDistributed: true }, { from: ["n_drop"] })
      .outputs("n_cls", "n_box")
      .build();
    const ir = build(wire, "n5-rcnn-cell43");
    const report = ir.verification.perModel.m!;

    expect(shape(ir, features)).toBe("(B, 28, 28, 128)");
    expect(shape(ir, "n_roi_pool")).toBe("(B, R, 7, 7, 128)");
    expect(shape(ir, "n_flatten")).toBe("(B, R, 6272)");
    expect(shape(ir, "n_cls")).toBe("(B, R, 9)");
    expect(params(ir, "n_fc")).toBe(802_944);
    expect(report.totals.computed.total).toBe(897_869);
    expect(report.totals.match).toBe(true);
    expect(report.rows.filter((row) => row.status !== "match")).toEqual([]);
    expect(report.unmatchedRows).toEqual([]);
    expect(problems(ir)).toEqual([]);
  });

  it("RPN: 246,637 params and 7,056 anchors", () => {
    const b = new WireBuilder({ title: "RPN", summary: "", framework: "keras", task: "detection" })
      .symbol("NUM_ANCHORS_PER_LOCATION", 9)
      .symbol("TOTAL_ANCHORS", 28 * 28 * 9, { provenance: "computed" })
      .model("m", "RPN", "", [ref(58, 1, 19, "definition")])
      .node("n_image", "input", { shape: [224, 224, 3] });
    const features = smallBackbone(b, "n_image");
    const wire = b
      .node("n_rpn_conv", "conv2d", relu(128), { from: [features] })
      .node(
        "n_obj_grid",
        "conv2d",
        { filters: "NUM_ANCHORS_PER_LOCATION", kernel: 1, activation: "sigmoid" },
        { from: ["n_rpn_conv"] },
      )
      .node("n_obj", "reshape", { shape: [7056, 1] }, { from: ["n_obj_grid"] })
      .node("n_box_grid", "conv2d", { filters: 36, kernel: 1 }, { from: ["n_rpn_conv"] })
      .node("n_box", "reshape", { shape: [-1, 4] }, { from: ["n_box_grid"] })
      .outputs("n_obj", "n_box")
      .build();
    const ir = build(wire, "n5-rcnn-cell58");
    const report = ir.verification.perModel.m!;

    expect(shape(ir, "n_obj")).toBe("(B, 7056, 1)");
    expect(shape(ir, "n_box")).toBe("(B, 7056, 4)");
    expect(report.totals.computed.total).toBe(246_637);
    expect(report.totals.match).toBe(true);
    expect(report.rows.filter((row) => row.status !== "match")).toEqual([]);
    expect(report.unmatchedRows).toEqual([]);
  });
});
