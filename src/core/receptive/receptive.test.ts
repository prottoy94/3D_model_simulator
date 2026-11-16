import { describe, expect, it } from "vitest";
import { modelStats, ratioText } from "../compare";
import { GALLERY } from "../gallery";
import { buildRuntimeIR, type RuntimeIR } from "../ir";
import { coverage, describeField, receptiveFields } from "./index";

function load(name: string): { ir: RuntimeIR; modelId: string } {
  const built = buildRuntimeIR(GALLERY[name]!(), { provenance: "user" });
  if (!built.ok) throw new Error(`${name} did not build`);
  return { ir: built.ir, modelId: built.ir.models[0]!.id };
}

/** The field of the last layer of the given op kind. */
function lastOf(ir: RuntimeIR, modelId: string, op: string) {
  const fields = receptiveFields(ir, modelId);
  const ids = (ir.index.topoOrder[modelId] ?? []).filter((id) => ir.nodes[id]!.op === op);
  return fields[ids[ids.length - 1]!] ?? null;
}

describe("receptive fields", () => {
  it("matches the textbook values for VGG16", () => {
    const { ir, modelId } = load("vgg16");
    const fields = receptiveFields(ir, modelId);
    const order = ir.index.topoOrder[modelId]!;
    const convs = order.filter((id) => ir.nodes[id]!.op === "conv2d");
    // 3×3 convolutions with stride 1: the first sees 3 px, the second 5 px.
    expect(fields[convs[0]!]!.size).toEqual([3, 3]);
    expect(fields[convs[1]!]!.size).toEqual([5, 5]);
    // After the fifth pooling layer: 212 px, one output step = 32 input px.
    const pool5 = lastOf(ir, modelId, "maxpool2d")!;
    expect(pool5.size).toEqual([212, 212]);
    expect(pool5.jump).toEqual([32, 32]);
    expect(pool5.global).toBe(false);
  });

  it("grows by (kernel − 1) × jump through AlexNet", () => {
    const { ir, modelId } = load("alexnet");
    const fields = receptiveFields(ir, modelId);
    const order = ir.index.topoOrder[modelId]!;
    let expected = 1;
    let jump = 1;
    for (const id of order) {
      const node = ir.nodes[id]!;
      if (node.op !== "conv2d" && node.op !== "maxpool2d") continue;
      const p = node.params as Record<string, [number, number] | undefined>;
      const kernel = (node.op === "conv2d" ? p.kernel : p.pool)![0];
      const stride = (p.stride ?? p.pool ?? [1, 1])[0];
      expected += (kernel - 1) * jump;
      jump *= stride;
      expect(fields[id]!.size[0]).toBe(expected);
      expect(fields[id]!.jump[0]).toBe(jump);
    }
    expect(expected).toBeGreaterThan(100);
  });

  it("is the whole image after flatten, and stays so", () => {
    const { ir, modelId } = load("alexnet");
    const dense = lastOf(ir, modelId, "dense")!;
    expect(dense.global).toBe(true);
    expect(coverage(dense)).toEqual([1, 1]);
    expect(describeField(dense)).toBe("whole image");
  });

  it("is one patch before attention and the whole image after it", () => {
    const { ir, modelId } = load("mini-vit");
    const fields = receptiveFields(ir, modelId);
    const order = ir.index.topoOrder[modelId]!;
    const attention = order.find((id) => ir.nodes[id]!.op === "multi_head_attention")!;
    expect(fields[attention]!.global).toBe(true);
    const patch = order.find((id) => ["patchify", "patch_embed"].includes(ir.nodes[id]!.op))!;
    const field = fields[patch]!;
    expect(field.global).toBe(false);
    expect(field.size[0]).toBe(field.jump[0]);
    expect(field.size[0]).toBeGreaterThan(1);
  });

  it("is unknown after a pretrained backbone", () => {
    const { ir, modelId } = load("firenet");
    const fields = receptiveFields(ir, modelId);
    const order = ir.index.topoOrder[modelId]!;
    const backbone = order.find((id) => ir.nodes[id]!.op === "backbone")!;
    expect(fields[backbone]).toBeNull();
    expect(fields[order[order.length - 1]!]).toBeNull();
  });
});

describe("model stats", () => {
  it("agrees with the engine's totals", () => {
    const { ir, modelId } = load("vgg16");
    const stats = modelStats(ir, modelId)!;
    expect(stats.params).toBe(14_981_448);
    expect(stats.families.reduce((sum, share) => sum + share.params, 0)).toBe(stats.params);
    expect(stats.families.reduce((sum, share) => sum + share.layers, 0)).toBe(stats.layers);
    expect(stats.depth).toBe(stats.layers);
    expect(stats.complete).toBe(true);
  });

  it("counts shared weights once", () => {
    const { ir, modelId } = load("firenet");
    const stats = modelStats(ir, modelId)!;
    expect(stats.params).toBe(32_183_742);
    expect(stats.families.reduce((sum, share) => sum + share.params, 0)).toBe(stats.params);
    expect(stats.depth).toBeLessThan(stats.layers);
  });

  it("words a ratio", () => {
    expect(ratioText(30, 10)).toBe("3.0× more");
    expect(ratioText(10, 300)).toBe("30× fewer");
    expect(ratioText(110, 100)).toBe("10% more");
    expect(ratioText(5, 5)).toBe("the same");
  });
});
