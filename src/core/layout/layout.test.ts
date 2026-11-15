import { describe, expect, it } from "vitest";
import { generateListing } from "../codegen/listing";
import { buildRuntimeIR } from "../ir/build";
import { WireBuilder } from "../ir/builder";
import { alexnetIR, miniVitIR, vgg16IR } from "../ir/examples";
import type { RuntimeIR } from "../ir/runtime";
import { buildVisibleGraph, defaultCollapsed } from "./graph";
import { layoutModel2D } from "./layout2d";
import { bezierPoint, glyphSize, layoutModel3D, type LayoutItem } from "./layout3d";

function build(wire: unknown): RuntimeIR {
  const result = buildRuntimeIR(wire);
  if (!result.ok) throw new Error(result.errors.join("\n"));
  return result.ir;
}

const NONE: ReadonlySet<string> = new Set();

function overlaps(a: LayoutItem, b: LayoutItem): boolean {
  return [0, 1, 2].every(
    (axis) =>
      Math.abs(a.position[axis]! - b.position[axis]!) < (a.size[axis]! + b.size[axis]!) / 2 - 1e-6,
  );
}

describe("visible graph", () => {
  it("keeps a plain chain on one lane with increasing ranks", () => {
    const graph = buildVisibleGraph(build(alexnetIR()), "m_alexnet", NONE);
    expect(graph.items).toHaveLength(17);
    expect(graph.items.map((item) => item.rank)).toEqual(Array.from({ length: 17 }, (_, i) => i));
    expect(new Set(graph.items.map((item) => item.lane))).toEqual(new Set([0]));
  });

  it("folds a collapsed group into one item and reroutes its edges", () => {
    const ir = build(vgg16IR());
    const graph = buildVisibleGraph(ir, "m_vgg16", new Set(["g_block3", "g_block4"]));
    const block = graph.items.find((item) => item.id === "g_block3")!;
    expect(block.kind).toBe("group");
    expect(block.memberIds).toHaveLength(4);
    expect(block.params).toBe(ir.groups.g_block3!.stats.params.trainable);
    expect(graph.items).toHaveLength(23 - 8 + 2);
    expect(graph.edges.some((edge) => edge.from === "n_pool2" && edge.to === "g_block3")).toBe(
      true,
    );
    expect(graph.edges.some((edge) => edge.from === "g_block3" && edge.to === "g_block4")).toBe(
      true,
    );
  });

  it("marks residual connections as skips and keeps the block on the main lane", () => {
    const graph = buildVisibleGraph(build(miniVitIR()), "m_vit", NONE);
    const residual = graph.edges.find((edge) => edge.from === "n_pos" && edge.to === "n_add1#1")!;
    expect(residual.kind).toBe("residual");
    expect(residual.span).toBe(3);
    expect(new Set(graph.items.map((item) => item.lane))).toEqual(new Set([0]));
  });

  it("spreads branches and extra inputs over lanes without collisions", () => {
    const wire = new WireBuilder({ title: "t", summary: "", framework: "keras" })
      .model("m", "Net")
      .node("n_image", "input", { shape: [32, 32, 3] })
      .node("n_rois", "input", { shape: [-1, 4], dynamicDim: "R" })
      .node("n_conv", "conv2d", { filters: 8, kernel: 3, padding: "same" }, { from: ["n_image"] })
      .node("n_pool", "roi_pool", { pooled: 7 }, { from: ["n_conv", "n_rois"] })
      .node("n_flat", "flatten", { timeDistributed: true }, { from: ["n_pool"] })
      .node("n_cls", "dense", { units: 9, timeDistributed: true }, { from: ["n_flat"] })
      .node("n_box", "dense", { units: 4, timeDistributed: true }, { from: ["n_flat"] })
      .build();
    const graph = buildVisibleGraph(build(wire), "m", NONE);
    const cells = graph.items.map((item) => `${item.rank}:${item.lane}`);
    expect(new Set(cells).size).toBe(cells.length);
    const lane = (id: string) => graph.items.find((item) => item.id === id)!.lane;
    expect(lane("n_image")).toBe(0);
    expect(lane("n_rois")).not.toBe(0);
    expect(lane("n_cls")).not.toBe(lane("n_box"));
  });

  it("starts with big groups collapsed", () => {
    expect(defaultCollapsed(build(alexnetIR()))).toEqual([]);
  });
});

describe("3D layout", () => {
  for (const [name, make, modelId] of [
    ["AlexNet", alexnetIR, "m_alexnet"],
    ["VGG16", vgg16IR, "m_vgg16"],
    ["mini-ViT", miniVitIR, "m_vit"],
  ] as const) {
    it(`${name}: no glyphs overlap, the flow runs along +X, frames enclose their members`, () => {
      const layout = layoutModel3D(build(make()), modelId, NONE);
      for (let i = 0; i < layout.items.length; i++) {
        for (let j = i + 1; j < layout.items.length; j++) {
          expect(
            overlaps(layout.items[i]!, layout.items[j]!),
            `${layout.items[i]!.id} / ${layout.items[j]!.id}`,
          ).toBe(false);
        }
      }
      const byId = new Map(layout.items.map((item) => [item.id, item]));
      for (const edge of layout.edges) {
        expect(byId.get(edge.to)!.position[0]).toBeGreaterThan(byId.get(edge.from)!.position[0]);
      }
      for (const frame of layout.frames) {
        const inside = layout.items.filter((item) => item.groupPath.includes(frame.groupId));
        expect(inside.length).toBeGreaterThan(0);
        for (const item of inside) {
          for (let axis = 0; axis < 3; axis++) {
            expect(item.position[axis]! + item.size[axis]! / 2).toBeLessThanOrEqual(
              frame.max[axis]! + 1e-6,
            );
            expect(item.position[axis]! - item.size[axis]! / 2).toBeGreaterThanOrEqual(
              Math.min(frame.min[axis]!, 0.02) - 1e-6,
            );
          }
        }
      }
      // Everything stands above the floor and the model is centred along the flow.
      expect(
        Math.min(...layout.items.map((item) => item.position[1] - item.size[1] / 2)),
      ).toBeGreaterThan(0);
      expect(Math.abs(layout.bounds.min[0] + layout.bounds.max[0])).toBeLessThan(0.5);
    });
  }

  it("sizes tensors by the logarithm of their dimensions", () => {
    const big = glyphSize("volume", { dims: [{ sym: "B" }, 224, 224, 3], layout: "NHWC" }).size;
    const small = glyphSize("volume", { dims: [{ sym: "B" }, 7, 7, 512], layout: "NHWC" }).size;
    expect(big[1]).toBeGreaterThan(small[1]); // taller: 224 vs 7
    expect(big[0]).toBeLessThan(small[0]); // thinner: 3 vs 512 channels
    expect(big[1] / small[1]).toBeLessThan(3); // but a 32× size gap stays readable
    expect(glyphSize("neuron-column", { dims: [{ sym: "B" }, 4096], layout: "NC" }).units).toBe(
      4096,
    );
    const nchw = glyphSize("volume", { dims: [{ sym: "B" }, 64, 112, 56], layout: "NCHW" }).size;
    expect(nchw[1]).toBeGreaterThan(nchw[2]); // height 112 > width 56
  });

  it("lifts residual edges above the tallest glyph", () => {
    const layout = layoutModel3D(build(miniVitIR()), "m_vit", NONE);
    const residual = layout.edges.find((edge) => edge.arc)!;
    const top = Math.max(...layout.items.map((item) => item.position[1] + item.size[1] / 2));
    expect(bezierPoint(residual.points, 0.5)[1]).toBeGreaterThan(top);
    const straight = layout.edges.find((edge) => !edge.arc)!;
    expect(bezierPoint(straight.points, 0)).toEqual(straight.points[0]);
  });

  it("nests the unrolled blocks inside the repeat frame", () => {
    const layout = layoutModel3D(build(miniVitIR()), "m_vit", NONE);
    const outer = layout.frames.find((frame) => frame.groupId === "g_blocks")!;
    const inner = layout.frames.find((frame) => frame.groupId === "g_blocks#2")!;
    expect(outer.level).toBe(1);
    expect(inner.level).toBe(0);
    expect(outer.min[0]).toBeLessThan(inner.min[0]);
    expect(outer.max[2]).toBeGreaterThan(inner.max[2]);
  });
});

describe("2D layout", () => {
  it("places cards top to bottom without overlap and frames around groups", () => {
    const layout = layoutModel2D(build(miniVitIR()), "m_vit", new Set(["g_blocks#3"]));
    const spots = layout.cards.map((card) => `${card.x}:${card.y}`);
    expect(new Set(spots).size).toBe(spots.length);
    for (const edge of layout.edges) {
      const from = layout.cards.find((card) => card.id === edge.from)!;
      const to = layout.cards.find((card) => card.id === edge.to)!;
      expect(to.y).toBeGreaterThan(from.y);
    }
    expect(layout.cards.some((card) => card.id === "g_blocks#3" && card.kind === "group")).toBe(
      true,
    );
    const outer = layout.frames.find((frame) => frame.groupId === "g_blocks")!;
    const inner = layout.frames.find((frame) => frame.groupId === "g_blocks#1")!;
    expect(outer.x).toBeLessThan(inner.x);
    expect(outer.width).toBeGreaterThan(inner.width);
    expect(layout.frames.indexOf(outer)).toBeLessThan(layout.frames.indexOf(inner));
  });
});

describe("generated listing", () => {
  it("gives every node of a gallery model its own line", () => {
    const ir = build(alexnetIR());
    const listing = generateListing(ir, "m_alexnet");
    const lines = listing.text.split("\n");
    expect(lines[listing.lineOfNode.n_conv1! - 1]).toBe(
      'layers.Conv2D(96, (11, 11), strides=(4, 4), padding="valid", activation="relu")  # → (B, 54, 54, 96)',
    );
    expect(new Set(Object.values(listing.lineOfNode)).size).toBe(17);
    const [first, last] = listing.linesOfGroup.g_block1!;
    expect(last - first).toBe(2);
    expect(lines[first - 2]).toBe("# ── Conv block 1 ──");
  });
});
