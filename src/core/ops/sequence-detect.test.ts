/** 1D convolution and pooling, and the detection steps after the heads. */
import { describe, expect, it } from "vitest";
import { tinyDetectorIR } from "../gallery";
import { buildRuntimeIR } from "../ir/build";
import { WireBuilder } from "../ir/builder";
import type { RuntimeIR } from "../ir/runtime";
import { shapeText } from "../shapes/tensor";
import { iou, suppress } from "../sim/boxes";
import { simulate, type NodeResult } from "../sim/run";

function build(wire: unknown): RuntimeIR {
  const result = buildRuntimeIR(wire);
  if (!result.ok) throw new Error(result.errors.join("\n"));
  return result.ir;
}
const model = (framework: "keras" | "pytorch") =>
  new WireBuilder({ title: "t", summary: "", framework }).model("m", "Net");
const params = (ir: RuntimeIR, id: string) =>
  ir.nodes[id]!.paramCount.trainable + ir.nodes[id]!.paramCount.nonTrainable;
const shape = (ir: RuntimeIR, id: string) => shapeText(ir.nodes[id]!.outShape!);
const errors = (ir: RuntimeIR) =>
  Object.values(ir.nodes)
    .filter((node) => node.status === "error")
    .map((node) => `${node.id}: ${node.issues[0]?.message}`);
const run = (ir: RuntimeIR) =>
  new Map<string, NodeResult>(
    [...simulate(ir, "m", { kind: "shapes" })].map((result) => [result.nodeId, result]),
  );
const finite = (data: Float32Array) => data.every((value) => Number.isFinite(value));

describe("1D convolution and pooling", () => {
  const keras = build(
    model("keras")
      .node("in", "input", { shape: [100, 32] })
      .node("conv", "conv1d", { filters: 64, kernel: 3 }, { from: ["in"] })
      .node(
        "same",
        "conv1d",
        { filters: 8, kernel: 5, stride: 2, padding: "same" },
        { from: ["in"] },
      )
      .node("pool", "maxpool1d", { pool: 2 }, { from: ["conv"] })
      .build(),
  );
  const torch = build(
    model("pytorch")
      .node("in", "input", { shape: [16, 100] })
      .node("conv", "conv1d", { filters: 32, kernel: 5, padding: 2 }, { from: ["in"] })
      .node("pool", "maxpool1d", { pool: 4 }, { from: ["conv"] })
      .node(
        "grouped",
        "conv1d",
        { filters: 32, kernel: 3, groups: 4, bias: false },
        { from: ["pool"] },
      )
      .build(),
  );

  it("follows Keras: (batch, length, channels)", () => {
    expect(errors(keras)).toEqual([]);
    expect(shape(keras, "conv")).toBe("(B, 98, 64)");
    expect(params(keras, "conv")).toBe(3 * 32 * 64 + 64); // 6,208
    expect(shape(keras, "same")).toBe("(B, 50, 8)");
    expect(shape(keras, "pool")).toBe("(B, 49, 64)");
    expect(params(keras, "pool")).toBe(0);
  });

  it("follows PyTorch: (batch, channels, length)", () => {
    expect(errors(torch)).toEqual([]);
    expect(shape(torch, "conv")).toBe("(B, 32, 100)");
    expect(params(torch, "conv")).toBe(5 * 16 * 32 + 32); // 2,592
    expect(shape(torch, "pool")).toBe("(B, 32, 25)");
    expect(shape(torch, "grouped")).toBe("(B, 32, 23)");
    expect(params(torch, "grouped")).toBe(3 * (32 / 4) * 32); // 768
  });

  it("is really computed in both layouts", () => {
    for (const ir of [keras, torch]) {
      const results = run(ir);
      for (const id of Object.keys(ir.nodes)) {
        const result = results.get(id)!;
        expect(result.tensor.illustrative, `${id}: ${result.note}`).toBe(false);
        expect(finite(result.tensor.data), id).toBe(true);
      }
    }
    const results = run(keras);
    const conv = results.get("conv")!.tensor;
    const pool = results.get("pool")!.tensor;
    expect(conv.shape).toEqual([98, 64]);
    expect(pool.shape).toEqual([49, 64]);
    // Pooling keeps the larger of each pair of steps, channel by channel.
    for (const [t, c] of [
      [0, 0],
      [7, 13],
      [48, 63],
    ] as const) {
      const expected = Math.max(conv.data[2 * t * 64 + c]!, conv.data[(2 * t + 1) * 64 + c]!);
      expect(pool.data[t * 64 + c]).toBe(expected);
    }
  });

  it("rejects an input that is not a sequence", () => {
    const bad = build(
      model("keras")
        .node("in", "input", { shape: [32, 32, 3] })
        .node("conv", "conv1d", { filters: 8, kernel: 3 }, { from: ["in"] })
        .build(),
    );
    expect(errors(bad)[0]).toContain("must be a sequence");
  });
});

describe("overlap and suppression", () => {
  it("computes intersection over union", () => {
    expect(iou([0, 0, 2, 2], [1, 1, 3, 3])).toBeCloseTo(1 / 7, 10);
    expect(iou([0, 0, 1, 1], [0, 0, 1, 1])).toBe(1);
    expect(iou([0, 0, 1, 1], [2, 2, 3, 3])).toBe(0);
  });

  it("keeps the best box of each overlapping cluster", () => {
    // A and B cover almost the same area; C is elsewhere.
    const boxes = Float32Array.from([0, 0, 1, 1, 0, 0, 1, 0.9, 2, 2, 3, 3]);
    const scores = [0.8, 0.9, 0.7];
    expect(suppress(boxes, scores, 0.5, 0, 100)).toEqual([1, 2]);
    // A looser overlap limit keeps all three, best first.
    expect(suppress(boxes, scores, 0.95, 0, 100)).toEqual([1, 0, 2]);
    expect(suppress(boxes, scores, 0.5, 0.75, 100)).toEqual([1]);
    expect(suppress(boxes, scores, 0.5, 0, 1)).toEqual([1]);
  });
});

describe("a detection pipeline after the heads", () => {
  const ir = build(
    model("keras")
      .node("in", "input", { shape: [64, 64, 3] })
      .node(
        "feat",
        "conv2d",
        { filters: 8, kernel: 3, stride: 4, padding: "same" },
        { from: ["in"] },
      )
      .node("anchors", "anchor_generator", { perLocation: 3 }, { from: ["feat"] })
      .node("box_head", "conv2d", { filters: 12, kernel: 1 }, { from: ["feat"] })
      .node("deltas", "reshape", { shape: [768, 4] }, { from: ["box_head"] })
      .node("cls_head", "conv2d", { filters: 6, kernel: 1 }, { from: ["feat"] })
      .node("scores", "reshape", { shape: [768, 2] }, { from: ["cls_head"] })
      .node("boxes", "box_decode", {}, { from: ["deltas", "anchors"] })
      .node("kept", "nms", { iouThreshold: 0.4, maxOutput: 50 }, { from: ["boxes", "scores"] })
      .node("crops", "roi_align", { pooled: 7 }, { from: ["feat", "kept"] })
      .build(),
  );

  it("has the shapes a detector has", () => {
    expect(errors(ir)).toEqual([]);
    expect(shape(ir, "anchors")).toBe("(B, 768, 4)");
    expect(shape(ir, "boxes")).toBe("(B, 768, 4)");
    // How many boxes survive depends on the picture.
    expect(shape(ir, "kept")).toBe("(B, K, 4)");
    expect(shape(ir, "crops")).toBe("(B, K, 7, 7, 8)");
    for (const id of ["boxes", "kept", "crops"]) expect(params(ir, id)).toBe(0);
  });

  it("decodes boxes inside the image and suppresses the overlapping ones", () => {
    const results = run(ir);
    const boxes = results.get("boxes")!.tensor;
    expect(boxes.illustrative).toBe(false);
    expect(boxes.data.every((value) => value >= 0 && value <= 1)).toBe(true);
    for (let i = 0; i < 768; i++) {
      expect(boxes.data[i * 4 + 2]!).toBeGreaterThanOrEqual(boxes.data[i * 4]!);
      expect(boxes.data[i * 4 + 3]!).toBeGreaterThanOrEqual(boxes.data[i * 4 + 1]!);
    }

    const kept = results.get("kept")!.tensor;
    const rows = kept.shape[0]!;
    expect(rows).toBeGreaterThan(1);
    for (let a = 0; a < rows; a++) {
      for (let b = a + 1; b < rows; b++) {
        const overlap = iou(
          kept.data.subarray(a * 4, a * 4 + 4),
          kept.data.subarray(b * 4, b * 4 + 4),
        );
        expect(overlap).toBeLessThanOrEqual(0.4);
      }
    }

    const crops = results.get("crops")!.tensor;
    expect(crops.shape).toEqual([rows, 7, 7, 8]);
    expect(crops.illustrative).toBe(false);
    expect(finite(crops.data)).toBe(true);
  });

  it("reports a count mismatch between offsets and anchors", () => {
    const bad = build(
      model("keras")
        .node("a", "input", { shape: [10, 4] })
        .node("b", "input", { shape: [12, 4] })
        .node("boxes", "box_decode", {}, { from: ["a", "b"] })
        .build(),
    );
    expect(errors(bad)[0]).toContain("10 predicted offsets but 12 anchors");
  });
});

describe("RoI align", () => {
  it("returns the feature map itself for a box that covers everything", () => {
    // With one sample point per cell and a full-image box, every cell centre is a pixel centre.
    const ir = build(
      model("keras")
        .node("in", "input", { shape: [6, 6, 2] })
        .node("rois", "input", { shape: [1, 4] })
        .node("crops", "roi_align", { pooled: 6, samplingRatio: 1 }, { from: ["in", "rois"] })
        .build(),
    );
    expect(shape(ir, "crops")).toBe("(B, 1, 6, 6, 2)");
    const results = run(ir);
    const boxes = results.get("rois")!.tensor.data;
    const whole = boxes[0] === 0 && boxes[1] === 0 && boxes[2] === 1 && boxes[3] === 1;
    const crops = results.get("crops")!.tensor;
    expect(finite(crops.data)).toBe(true);
    if (whole) expect([...crops.data]).toEqual([...results.get("in")!.tensor.data]);
  });
});

describe("the Tiny detector gallery model", () => {
  const built = buildRuntimeIR(tinyDetectorIR(), { provenance: "user" });
  if (!built.ok) throw new Error(built.errors.join("\n"));
  const ir = built.ir;

  it("builds without problems and has the expected sizes", () => {
    expect(errors(ir)).toEqual([]);
    expect(Object.values(ir.nodes).every((node) => node.status === "ok")).toBe(true);
    expect(shape(ir, "n_conv3")).toBe("(B, 16, 16, 64)");
    expect(shape(ir, "n_anchors")).toBe("(B, 768, 4)");
    expect(shape(ir, "n_kept")).toBe("(B, K, 4)");
    expect(shape(ir, "n_crops")).toBe("(B, K, 7, 7, 64)");
    // 3·3·3·16+16 + 3·3·16·32+32 + 3·3·32·64+64 + 64·12+12 + 64·6+6
    expect(ir.verification.perModel.m_detector!.totals.computed.total).toBe(
      448 + 4_640 + 18_496 + 780 + 390,
    );
  });

  it("is computed from end to end", () => {
    const results = new Map(
      [...simulate(ir, "m_detector", { kind: "shapes" })].map((result) => [result.nodeId, result]),
    );
    for (const id of Object.keys(ir.nodes)) {
      expect(results.get(id)!.tensor.illustrative, `${id}: ${results.get(id)!.note}`).toBe(false);
    }
  });
});
