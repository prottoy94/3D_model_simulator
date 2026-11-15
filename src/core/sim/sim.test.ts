import { describe, expect, it } from "vitest";
import { buildRuntimeIR } from "../ir/build";
import { WireBuilder } from "../ir/builder";
import { alexnetIR, miniVitIR } from "../ir/examples";
import type { RuntimeIR } from "../ir/runtime";
import { buildVisibleGraph } from "../layout/graph";
import { buildPlan, focusPlan, stateAt, timeOfStep } from "./plan";
import { estimateWork, simulate, type NodeResult } from "./run";
import { canonicalAxis, canonicalShape, sizeOf } from "./tensor";
import { summarize, transferables } from "./view";

function build(wire: unknown): RuntimeIR {
  const result = buildRuntimeIR(wire);
  if (!result.ok) throw new Error(result.errors.join("\n"));
  return result.ir;
}

function run(
  ir: RuntimeIR,
  modelId: string,
  kind: "shapes" | "noise" = "shapes",
): Map<string, NodeResult> {
  const results = new Map<string, NodeResult>();
  for (const result of simulate(ir, modelId, { kind })) {
    // Keep a copy: the runner reuses nothing, but drops its own references.
    results.set(result.nodeId, result);
  }
  return results;
}

const keras = () =>
  new WireBuilder({ title: "t", summary: "", framework: "keras" }).model("m", "Net");

describe("kernels", () => {
  const ir = build(
    keras()
      .node("in", "input", { shape: [8, 8, 3] })
      .node(
        "conv",
        "conv2d",
        { filters: 4, kernel: 3, padding: "same", activation: "relu" },
        { from: ["in"] },
      )
      .node("pool", "maxpool2d", { pool: 2 }, { from: ["conv"] })
      .node("avg", "avgpool2d", { pool: 2 }, { from: ["conv"] })
      .node("bn", "batchnorm", {}, { from: ["pool"] })
      .node("gap", "global_avg_pool", {}, { from: ["bn"] })
      .node("fc", "dense", { units: 5, activation: "softmax" }, { from: ["gap"] })
      .node("flat", "flatten", {}, { from: ["pool"] })
      .node("cat", "concat", {}, { from: ["pool", "avg"] })
      // A per-channel gate keeps its two spatial axes (size 1) so it can scale the feature map.
      .node("squeeze", "global_avg_pool", { keepDims: true }, { from: ["bn"] })
      .node("gate", "sigmoid", {}, { from: ["squeeze"] })
      .node("scaled", "multiply", {}, { from: ["pool", "gate"] })
      .node("sum", "add", {}, { from: ["pool", "scaled"] })
      .build(),
  );
  const results = run(ir, "m");
  const data = (id: string) => results.get(id)!.tensor;

  it("produces every tensor in the shape the engine computed", () => {
    for (const [id, result] of results) {
      expect(result.tensor.data.length, id).toBe(sizeOf(canonicalShape(ir.nodes[id]!.outShape!)));
      expect(result.tensor.illustrative, id).toBe(false);
      expect([...result.tensor.data].every(Number.isFinite), id).toBe(true);
    }
  });

  it("relu leaves no negatives; pooling picks the max and the mean of each window", () => {
    const conv = data("conv");
    expect(Math.min(...conv.data)).toBeGreaterThanOrEqual(0);
    const window = [0, 1, 8, 9].map((offset) => conv.data[offset * 4]!); // channel 0, rows 0-1, cols 0-1
    expect(data("pool").data[0]).toBeCloseTo(Math.max(...window), 6);
    expect(data("avg").data[0]).toBeCloseTo(window.reduce((a, b) => a + b, 0) / 4, 6);
  });

  it("softmax gives probabilities; global pooling gives channel means", () => {
    const probabilities = data("fc").data;
    expect([...probabilities].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    const bn = data("bn");
    let mean = 0;
    for (let p = 0; p < 16; p++) mean += bn.data[p * 4 + 2]!;
    expect(data("gap").data[2]).toBeCloseTo(mean / 16, 5);
  });

  it("flatten keeps the values, concat joins channels, multiply broadcasts a per-channel gate", () => {
    expect([...data("flat").data]).toEqual([...data("pool").data]);
    const cat = data("cat");
    expect(cat.shape).toEqual([4, 4, 8]);
    expect(cat.data[5]).toBe(data("avg").data[1]); // position 0, channel 5 = avg channel 1
    const gate = data("gate").data;
    expect(data("scaled").data[4 + 3]).toBeCloseTo(data("pool").data[4 + 3]! * gate[3]!, 6);
    expect(data("sum").data[9]).toBeCloseTo(data("pool").data[9]! + data("scaled").data[9]!, 6);
  });

  it("is deterministic, and the input changes the result", () => {
    const again = run(ir, "m");
    expect([...again.get("fc")!.tensor.data]).toEqual([...data("fc").data]);
    const noise = run(ir, "m", "noise");
    expect([...noise.get("fc")!.tensor.data]).not.toEqual([...data("fc").data]);
  });

  it("gives different weights to different layers of the same size", () => {
    const twin = build(
      keras()
        .node("in", "input", { shape: [6] })
        .node("a", "dense", { units: 3 }, { from: ["in"] })
        .node("b", "dense", { units: 3 }, { from: ["in"] })
        .build(),
    );
    const out = run(twin, "m");
    expect([...out.get("a")!.tensor.data]).not.toEqual([...out.get("b")!.tensor.data]);
  });
});

describe("layouts and wrappers", () => {
  it("runs a PyTorch (channels-first) model and maps axes to channel-last", () => {
    const ir = build(
      new WireBuilder({ title: "t", summary: "", framework: "pytorch" })
        .model("m", "Net")
        .node("in", "input", { shape: [3, 16, 12] })
        .node("conv", "conv2d", { filters: 8, kernel: 3, stride: 2, padding: 1 }, { from: ["in"] })
        .node("cat", "concat", { axis: 1 }, { from: ["conv", "conv"] })
        .node("gap", "global_avg_pool", {}, { from: ["cat"] })
        .node("flat", "flatten", {}, { from: ["gap"] })
        .node("fc", "dense", { units: 2 }, { from: ["flat"] })
        .build(),
    );
    expect(canonicalShape(ir.nodes.conv!.outShape!)).toEqual([8, 6, 8]);
    expect(canonicalAxis(1, ir.nodes.conv!.outShape!)).toBe(2);
    const results = run(ir, "m");
    expect(results.get("cat")!.tensor.shape).toEqual([8, 6, 16]);
    expect(results.get("fc")!.tensor.data).toHaveLength(2);
    expect([...results.values()].every((result) => !result.tensor.illustrative)).toBe(true);
  });

  it("applies TimeDistributed layers per region and pools regions from the feature map", () => {
    const ir = build(
      keras()
        .node("image", "input", { shape: [16, 16, 3] })
        .node("rois", "input", { shape: [-1, 4], dynamicDim: "R" })
        .node("conv", "conv2d", { filters: 6, kernel: 3, padding: "same" }, { from: ["image"] })
        .node("pool", "roi_pool", { pooled: 3 }, { from: ["conv", "rois"] })
        .node("flat", "flatten", { timeDistributed: true }, { from: ["pool"] })
        .node(
          "cls",
          "dense",
          { units: 4, activation: "softmax", timeDistributed: true },
          { from: ["flat"] },
        )
        .build(),
    );
    const results = run(ir, "m");
    expect(results.get("pool")!.tensor.shape).toEqual([4, 3, 3, 6]);
    const cls = results.get("cls")!.tensor;
    expect(cls.shape).toEqual([4, 4]);
    for (let roi = 0; roi < 4; roi++) {
      expect([...cls.data.subarray(roi * 4, roi * 4 + 4)].reduce((a, b) => a + b, 0)).toBeCloseTo(
        1,
        5,
      );
    }
  });

  it("marks a pretrained backbone and everything after it as illustrative", () => {
    const ir = build(
      keras()
        .node("in", "input", { shape: [64, 64, 3] })
        .node("base", "backbone", { outputShape: [2, 2, 32], paramCount: 10 }, { from: ["in"] })
        .node("gap", "global_avg_pool", {}, { from: ["base"] })
        .node("fc", "dense", { units: 3 }, { from: ["gap"] })
        .build(),
    );
    const results = run(ir, "m");
    expect(results.get("in")!.tensor.illustrative).toBe(false);
    expect(results.get("base")!.tensor.illustrative).toBe(true);
    expect(results.get("base")!.note).toContain("pretrained network");
    expect(results.get("fc")!.tensor.illustrative).toBe(true);
    expect(results.get("fc")!.note).toContain("illustrative");
  });
});

describe("whole models", () => {
  it("runs the mini-ViT end to end with real attention", () => {
    const ir = build(miniVitIR());
    const results = run(ir, "m_vit");
    expect(results.size).toBe(46);
    expect([...results.values()].every((result) => !result.tensor.illustrative)).toBe(true);

    const attention = results.get("n_mha#1")!.extras.attention!;
    expect(attention.heads).toBe(4);
    expect(attention.tokens).toBe(196);
    // Each token's attention over the 196 tokens sums to 1.
    expect([...attention.data.subarray(0, 196)].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4);
    // The four blocks have their own weights.
    expect([...results.get("n_mha#2")!.tensor.data.subarray(0, 8)]).not.toEqual([
      ...results.get("n_mha#1")!.tensor.data.subarray(0, 8),
    ]);
    const output = results.get("n_out")!.tensor.data;
    expect(output).toHaveLength(8);
    expect([...output].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
  }, 60_000);

  it("estimates the work before running", () => {
    const alexnet = build(alexnetIR());
    const work = estimateWork(alexnet, "m_alexnet");
    expect(work).toBeGreaterThan(0.9e9);
    expect(work).toBeLessThan(1.3e9);
  });
});

describe("views", () => {
  const ir = build(miniVitIR());
  const views = new Map<string, ReturnType<typeof summarize>>();
  for (const result of simulate(ir, "m_vit", { kind: "shapes" })) {
    if (["n_in", "n_patches", "n_mha#1", "n_gap", "n_out"].includes(result.nodeId)) {
      views.set(result.nodeId, summarize(result, ir.nodes[result.nodeId]!.op === "input"));
    }
  }

  it("shows the input as a colour picture and tokens as a heatmap", () => {
    const input = views.get("n_in")!;
    expect(input.kind).toBe("image");
    expect(input.picture).toMatchObject({ width: 128, height: 128 });
    expect(input.stats.min).toBeGreaterThanOrEqual(0);
    const patches = views.get("n_patches")!;
    expect(patches.kind).toBe("tokens");
    expect(patches.picture).toMatchObject({ width: 128, height: 128 });
  });

  it("carries attention maps, vectors and probabilities", () => {
    const attention = views.get("n_mha#1")!.attention!;
    expect(attention.pictures).toHaveLength(4);
    expect(attention.pictures[0]).toMatchObject({ width: 96, height: 96 });
    expect(views.get("n_gap")!.values).toHaveLength(64);
    expect(views.get("n_gap")!.probabilities).toBe(false);
    expect(views.get("n_out")!.probabilities).toBe(true);
    expect(views.get("n_out")!.histogram.reduce((a, b) => a + b, 0)).toBe(8);
    expect(transferables(views.get("n_mha#1")!).length).toBeGreaterThan(4);
  });

  it("builds a channel mosaic for feature maps", () => {
    const cnn = build(
      keras()
        .node("in", "input", { shape: [16, 16, 3] })
        .node("conv", "conv2d", { filters: 10, kernel: 3 }, { from: ["in"] })
        .build(),
    );
    let view: ReturnType<typeof summarize> | undefined;
    for (const result of simulate(cnn, "m", { kind: "stripes" })) {
      if (result.nodeId === "conv") view = summarize(result, false);
    }
    expect(view!.tiles).toEqual({ shown: 10, total: 10, columns: 4 });
    expect(view!.picture).toMatchObject({ width: 4 * 15 - 1, height: 3 * 15 - 1 });
    expect(view!.kernel).toMatchObject({ height: 3, width: 3 });
    expect(view!.focus.input).toMatchObject({ width: 16, height: 16 });
    expect(view!.focus.output).toMatchObject({ width: 14, height: 14 });
  });
});

describe("plan", () => {
  const ir = build(miniVitIR());

  it("has one step per block, in flow order, and is a pure function of time", () => {
    const plan = buildPlan(buildVisibleGraph(ir, "m_vit", new Set()));
    expect(plan.steps).toHaveLength(46);
    expect(plan.steps[0]!.sources).toEqual([]);
    expect(plan.steps.find((step) => step.itemId === "n_add1#1")!.sources).toEqual([
      "n_mha#1",
      "n_pos",
    ]);
    for (const step of plan.steps.slice(1)) {
      expect(step.start).toBeCloseTo(
        plan.steps[step.index - 1]!.start + plan.steps[step.index - 1]!.duration,
        9,
      );
    }

    const third = plan.steps[3]!;
    const state = stateAt(plan, third.start + third.duration * 0.5);
    expect(state).toMatchObject({ step: 3, phase: "compute", ended: false });
    expect(state.progress).toBeCloseTo(0.5, 9);
    expect(stateAt(plan, third.start + third.duration * 0.1).phase).toBe("travel");
    expect(stateAt(plan, third.start + third.duration * 0.9).phase).toBe("emit");
    expect(stateAt(plan, -5).step).toBe(0);
    expect(stateAt(plan, plan.total + 1)).toMatchObject({ step: 45, ended: true, progress: 1 });
    expect(timeOfStep(plan, 3)).toBe(third.start);
    expect(stateAt(plan, timeOfStep(plan, 7)).step).toBe(7);
  });

  it("turns a collapsed group into one longer step", () => {
    const plan = buildPlan(buildVisibleGraph(ir, "m_vit", new Set(["g_blocks#1", "g_blocks#2"])));
    expect(plan.steps).toHaveLength(46 - 18 + 2);
    const block = plan.steps.find((step) => step.itemId === "g_blocks#1")!;
    expect(block.memberIds).toHaveLength(9);
    expect(block.duration).toBeGreaterThan(plan.steps[1]!.duration);
    expect(plan.steps.find((step) => step.itemId === "g_blocks#2")!.sources).toEqual([
      "g_blocks#1",
    ]);
  });

  it("keeps only the focused layers' steps for a Lens focus, re-timed from zero", () => {
    const whole = buildPlan(buildVisibleGraph(ir, "m_vit", new Set()));
    const block = whole.steps
      .filter((step) => step.itemId.endsWith("#2"))
      .map((step) => step.itemId);
    const plan = focusPlan(whole, block);
    expect(plan.steps.map((step) => step.itemId)).toEqual(block);
    expect(plan.steps).toHaveLength(9);
    expect(plan.steps[0]).toMatchObject({ index: 0, start: 0 });
    expect(plan.total).toBeCloseTo(
      plan.steps.reduce((sum, step) => sum + step.duration, 0),
      9,
    );
    expect(stateAt(plan, plan.total).ended).toBe(true);
    // A collapsed group that contains a focused layer stays as one step.
    const collapsed = buildPlan(buildVisibleGraph(ir, "m_vit", new Set(["g_blocks#2"])));
    expect(focusPlan(collapsed, block).steps.map((step) => step.itemId)).toEqual(["g_blocks#2"]);
  });
});
