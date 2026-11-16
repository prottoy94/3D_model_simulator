/**
 * Phase 8 layers: parameter formulas, shapes, compute, and the FireNet golden
 * total from docs/07 (the notebook prints 32.18 M parameters and 9,261 anchors).
 */
import { describe, expect, it } from "vitest";
import { draftToWire } from "../ai/draft";
import { fireNetDraft } from "../gallery";
import { OP_REFERENCE } from "../ai/prompt";
import { buildRuntimeIR } from "../ir/build";
import { WireBuilder } from "../ir/builder";
import type { RuntimeIR } from "../ir/runtime";
import { shapeText } from "../shapes/tensor";
import { simulate, type NodeResult } from "../sim/run";
import { CATALOG_OPS, getOp } from "./index";

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
function run(ir: RuntimeIR): Map<string, NodeResult> {
  return new Map(
    [...simulate(ir, "m", { kind: "shapes" })].map((result) => [result.nodeId, result]),
  );
}

describe("catalog", () => {
  it("has a complete entry and a prompt line for every op", () => {
    expect(CATALOG_OPS.length).toBe(50);
    for (const op of CATALOG_OPS) {
      const spec = getOp(op)!;
      expect(spec.title.length, op).toBeGreaterThan(0);
      expect(typeof spec.explain, op).toBe("function");
      expect(OP_REFERENCE[op], `prompt reference for ${op}`).toBeTruthy();
    }
  });
});

describe("recurrent layers", () => {
  const keras = build(
    model("keras")
      .node("in", "input", { shape: [20, 32] })
      .node("rnn", "rnn", { units: 64 }, { from: ["in"] })
      .node("lstm", "lstm", { units: 64, returnSequences: true }, { from: ["in"] })
      .node("gru", "gru", { units: 64 }, { from: ["in"] })
      .node("bi", "lstm", { units: 64, bidirectional: true }, { from: ["in"] })
      .build(),
  );

  it("counts parameters as Keras does", () => {
    expect(params(keras, "rnn")).toBe(64 * (32 + 64) + 64); // 6,208
    expect(params(keras, "lstm")).toBe(4 * (64 * (32 + 64) + 64)); // 24,832
    expect(params(keras, "gru")).toBe(3 * (64 * (32 + 64) + 2 * 64)); // 18,816
    expect(params(keras, "bi")).toBe(2 * 24_832);
    expect(shape(keras, "lstm")).toBe("(B, 20, 64)");
    expect(shape(keras, "gru")).toBe("(B, 64)");
    expect(shape(keras, "bi")).toBe("(B, 128)");
  });

  it("counts PyTorch's second bias and returns the whole sequence by default", () => {
    const torch = build(
      model("pytorch")
        .node("in", "input", { shape: [20, 32] })
        .node("lstm", "lstm", { units: 64 }, { from: ["in"] })
        .build(),
    );
    expect(params(torch, "lstm")).toBe(4 * (64 * (32 + 64) + 2 * 64)); // 25,088
    expect(shape(torch, "lstm")).toBe("(B, 20, 64)");
  });

  it("computes bounded, repeatable memories", () => {
    const results = run(keras);
    for (const id of ["rnn", "lstm", "gru", "bi"]) {
      const data = results.get(id)!.tensor.data;
      expect(Math.max(...data.map(Math.abs)), id).toBeLessThanOrEqual(1);
      expect(
        [...data].some((value) => value !== 0),
        id,
      ).toBe(true);
    }
    expect(results.get("lstm")!.tensor.shape).toEqual([20, 64]);
    expect([...run(keras).get("gru")!.tensor.data]).toEqual([...results.get("gru")!.tensor.data]);
  });
});

describe("attention, norm and resizing", () => {
  const ir = build(
    model("pytorch")
      .node("in", "input", { shape: [64, 8, 8] })
      .node("gn", "groupnorm", { groups: 32 }, { from: ["in"] })
      .node("se", "squeeze_excite", { reduction: 16 }, { from: ["in"] })
      .node("ca", "channel_attention", { reduction: 16, minHidden: 8 }, { from: ["in"] })
      .node("sa", "spatial_attention", { kernel: 7 }, { from: ["in"] })
      .node("up", "upsample", { scale: 2 }, { from: ["in"] })
      .node("to", "upsample", { size: [28, 28], mode: "bilinear" }, { from: ["in"] })
      .node("ada", "adaptive_avg_pool2d", { output: 1 }, { from: ["in"] })
      .node("gap", "global_avg_pool", { keepDims: false }, { from: ["in"] })
      .node("gmp", "global_max_pool", { keepDims: false }, { from: ["in"] })
      .node("mean", "reduce_mean", { axis: [2, 3] }, { from: ["in"] })
      .node("amax", "reduce_max", { axis: 1, keepDims: true }, { from: ["in"] })
      .node("silu", "silu", {}, { from: ["in"] })
      .build(),
  );

  it("counts parameters and shapes", () => {
    expect(errors(ir)).toEqual([]);
    expect(params(ir, "gn")).toBe(128);
    expect(params(ir, "se")).toBe(64 * 4 + 4 + 4 * 64 + 64); // biases on
    expect(params(ir, "ca")).toBe(64 * 8 + 8 * 64); // hidden = max(64 // 16, 8), no biases
    expect(params(ir, "sa")).toBe(7 * 7 * 2);
    expect(shape(ir, "up")).toBe("(B, 64, 16, 16)");
    expect(shape(ir, "to")).toBe("(B, 64, 28, 28)");
    expect(shape(ir, "ada")).toBe("(B, 64, 1, 1)");
    expect(shape(ir, "gmp")).toBe("(B, 64)");
    expect(shape(ir, "mean")).toBe("(B, 64)");
    expect(shape(ir, "amax")).toBe("(B, 1, 8, 8)");
  });

  it("computes them: pooling variants agree, gates only shrink values, upsampling repeats", () => {
    const results = run(ir);
    const data = (id: string) => results.get(id)!.tensor.data;
    const input = data("in");
    expect([...results.values()].every((result) => !result.tensor.illustrative)).toBe(true);

    // Three ways to average over the image give the same 64 numbers.
    for (let c = 0; c < 64; c += 9) {
      expect(data("ada")[c]).toBeCloseTo(data("gap")[c]!, 5);
      expect(data("mean")[c]).toBeCloseTo(data("gap")[c]!, 5);
    }
    // The maximum over channels at position 0, and over positions for channel 0.
    expect(data("amax")[0]).toBeCloseTo(Math.max(...input.subarray(0, 64)), 6);
    let best = -Infinity;
    for (let pos = 0; pos < 64; pos++) best = Math.max(best, input[pos * 64]!);
    expect(data("gmp")[0]).toBeCloseTo(best, 6);

    // Attention multiplies by a number between 0 and 1.
    for (const id of ["se", "ca", "sa"]) {
      for (let i = 0; i < input.length; i += 97)
        expect(Math.abs(data(id)[i]!), id).toBeLessThanOrEqual(Math.abs(input[i]!) + 1e-6);
    }
    // Nearest upsampling by 2: output (y, x) reads input (y/2, x/2).
    expect(data("up")[(3 * 16 + 5) * 64 + 7]).toBe(input[(1 * 8 + 2) * 64 + 7]);
    // Group norm centres each group.
    let sum = 0;
    for (let pos = 0; pos < 64; pos++) sum += data("gn")[pos * 64]! + data("gn")[pos * 64 + 1]!;
    expect(sum / 128).toBeCloseTo(0, 4);
  });
});

describe("shared weights", () => {
  it("counts a shared layer once and computes it with the same weights", () => {
    const ir = build(
      model("keras")
        .node("in", "input", { shape: [16] })
        .node("head", "dense", { units: 4 }, { from: ["in"] })
        .node("again", "dense", { units: 4, sharedWith: "head" }, { from: ["in"] })
        .node("other", "dense", { units: 4 }, { from: ["in"] })
        .build(),
    );
    expect(params(ir, "head")).toBe(68);
    expect(params(ir, "again")).toBe(0);
    expect(ir.nodes.again!.issues[0]!.code).toBe("shared-weights");
    expect(ir.verification.perModel.m!.totals.computed.total).toBe(136);
    const results = run(ir);
    expect([...results.get("again")!.tensor.data]).toEqual([...results.get("head")!.tensor.data]);
    expect([...results.get("other")!.tensor.data]).not.toEqual([
      ...results.get("head")!.tensor.data,
    ]);
  });
});

// ── N6: FireNet ───────────────────────────────────────────────────────

describe("N6 FireNet (hand-written from the notebook's code)", () => {
  const ir = build(draftToWire(fireNetDraft()));
  const total = ir.verification.perModel.m_1!.totals.computed.total;

  it("has no shape errors and the feature sizes the notebook describes", () => {
    expect(errors(ir)).toEqual([]);
    expect(shape(ir, "f2")).toBe("(B, 192, 28, 28)");
    expect(shape(ir, "f3")).toBe("(B, 384, 14, 14)");
    expect(shape(ir, "base")).toBe("(B, 768, 7, 7)");
    expect(shape(ir, "p5")).toBe("(B, 256, 7, 7)");
    expect(shape(ir, "cat")).toBe("(B, 768, 28, 28)");
    expect(shape(ir, "both")).toBe("(B, 512)");
    expect(shape(ir, "cls_logits")).toBe("(B, 2)");
  });

  it("totals 32,183,742 parameters, the 32.18 M the notebook prints", () => {
    expect(params(ir, "base")).toBe(27_818_592);
    expect(params(ir, "f2")).toBe(0);
    expect(params(ir, "ca3")).toBe(8_192);
    expect(params(ir, "p3")).toBe(98);
    expect(params(ir, "cls_conv1_3")).toBe(589_824);
    expect(params(ir, "cls_conv1_4")).toBe(0); // the same tower, applied to the next scale
    expect(total).toBe(32_183_742);
    expect((total / 1e6).toFixed(2)).toBe("32.18");
  });

  it("produces 9,261 anchors and matching detection outputs", () => {
    expect(shape(ir, "anchors")).toBe("(B, 9261, 4)");
    expect(shape(ir, "det_cls")).toBe("(B, 9261, 2)");
    expect(shape(ir, "det_box")).toBe("(B, 9261, 4)");
    expect(28 * 28 * 9 + 14 * 14 * 9 + 7 * 7 * 9).toBe(9_261);
  });
});
