/**
 * The 3D "data river": data flows along +X, branches spread along Z, and every
 * tensor is a box whose size follows the logarithm of its dimensions, so a
 * 224×224 image and a 7×7 feature map both stay readable in one scene.
 */
import type { RuntimeIR } from "../ir/runtime";
import { isKnown, type TensorShape } from "../shapes/tensor";
import type { GlyphKind } from "../ops/types";
import { buildVisibleGraph, type VisibleEdge, type VisibleItem } from "./graph";

export type Vec3 = [number, number, number];

export type LayoutItem = VisibleItem & {
  /** Centre of the glyph. */
  position: Vec3;
  /** Thickness along the flow (x), height (y), depth (z). */
  size: Vec3;
  /** Vector length for neuron columns, when the output is a plain vector. */
  units: number | null;
};

export type LayoutEdge = VisibleEdge & {
  /** True when the edge is drawn as an arc above the river (residual or skip). */
  arc: boolean;
  /** Cubic Bézier control points. */
  points: [Vec3, Vec3, Vec3, Vec3];
};

export type LayoutFrame = {
  groupId: string;
  label: string;
  kind: string;
  /** 0 for the innermost frames; parents are one higher than their deepest child. */
  level: number;
  min: Vec3;
  max: Vec3;
};

export type ModelLayout = {
  items: LayoutItem[];
  edges: LayoutEdge[];
  frames: LayoutFrame[];
  bounds: { min: Vec3; max: Vec3 };
};

const GAP_X = 0.6;
const LANE_GAP = 1.0;
const FLOOR_CLEARANCE = 0.45;
const MIN_SIZE = 0.14;
const THIN = 0.09;
const UNKNOWN_DIM = 8;

const log = (value: number) => Math.log2(value + 1);

/** Glyphs that only touch values in place are drawn as thin plates. */
const PLATES: ReadonlySet<GlyphKind> = new Set(["plate", "ghost"]);

export function glyphSize(
  glyph: GlyphKind,
  shape: TensorShape | null,
  isGroup = false,
): { size: Vec3; units: number | null } {
  if (glyph === "merge") return { size: [0.42, 0.42, 0.42], units: null };
  const dims = (shape?.dims.slice(1) ?? []).map((dim) => (isKnown(dim) ? dim : UNKNOWN_DIM));
  const thin = PLATES.has(glyph);
  let size: Vec3;
  let units: number | null = null;

  if (dims.length >= 3) {
    // Image-like: the last three dims are height, width, channels (or channels first).
    const [a, b, c] = dims.slice(-3) as [number, number, number];
    const channelsFirst = shape?.layout === "NCHW";
    const h = channelsFirst ? b : a;
    const w = channelsFirst ? c : b;
    const channels = channelsFirst ? a : c;
    size = [thin ? THIN : 0.11 * log(channels), 0.3 * log(h), 0.3 * log(w)];
  } else if (dims.length === 2) {
    // A token sequence: tall by token count, deep by token width.
    const [tokens, width] = dims as [number, number];
    size = [thin ? THIN : 0.16, 0.3 * log(tokens), 0.16 * log(width)];
    if (glyph === "ring") size = [0.22, 0.3 * log(tokens), 0.3 * log(tokens)];
  } else if (dims.length === 1) {
    units = dims[0]!;
    size = [thin ? THIN : 0.3, 0.3 * log(units), thin ? 0.3 : 0.3];
  } else {
    size = [0.5, 0.5, 0.5];
  }
  if (isGroup) size = [Math.max(size[0], 0.5) * 1.6, size[1] * 1.08, size[2] * 1.08];
  return {
    size: size.map((value) => Math.max(MIN_SIZE, value)) as Vec3,
    units: isGroup ? null : units,
  };
}

export function layoutModel3D(
  ir: RuntimeIR,
  modelId: string,
  collapsed: ReadonlySet<string>,
): ModelLayout {
  const graph = buildVisibleGraph(ir, modelId, collapsed);
  const sized = graph.items.map((item) => ({
    item,
    ...glyphSize(item.glyph, item.outShape, item.kind === "group"),
  }));

  // Each column is as wide as its thickest glyph.
  const columnWidth = Array.from({ length: graph.ranks }, () => MIN_SIZE);
  let tallest = 0;
  let deepest = 0;
  for (const { item, size } of sized) {
    columnWidth[item.rank] = Math.max(columnWidth[item.rank]!, size[0]);
    tallest = Math.max(tallest, size[1]);
    deepest = Math.max(deepest, size[2]);
  }
  const columnX: number[] = [];
  let cursor = 0;
  for (const width of columnWidth) {
    columnX.push(cursor + width / 2);
    cursor += width + GAP_X;
  }
  const length = Math.max(0, cursor - GAP_X);
  const centreY = tallest / 2 + FLOOR_CLEARANCE;
  const laneStep = deepest + LANE_GAP;

  const items: LayoutItem[] = sized.map(({ item, size, units }) => ({
    ...item,
    size,
    units,
    // Centre the model on the origin along the flow so the camera can frame it.
    position: [columnX[item.rank]! - length / 2, centreY, item.lane * laneStep],
  }));
  const byId = new Map(items.map((item) => [item.id, item]));

  const edges: LayoutEdge[] = graph.edges.map((edge) => {
    const from = byId.get(edge.from)!;
    const to = byId.get(edge.to)!;
    const start: Vec3 = [from.position[0] + from.size[0] / 2, from.position[1], from.position[2]];
    const end: Vec3 = [to.position[0] - to.size[0] / 2, to.position[1], to.position[2]];
    const dx = Math.max(0.2, end[0] - start[0]);
    const arc = edge.kind === "residual" || edge.kind === "skip" || edge.span > 1;
    if (!arc) {
      return {
        ...edge,
        arc,
        points: [
          start,
          [start[0] + dx / 2, start[1], start[2]],
          [end[0] - dx / 2, end[1], end[2]],
          end,
        ],
      };
    }
    // Leave from the top of the producer, clear everything in between, land on top of the consumer.
    const top = centreY + tallest / 2 + 0.35 + 0.18 * Math.min(edge.span, 8);
    const lift: Vec3 = [from.position[0], from.position[1] + from.size[1] / 2, from.position[2]];
    const land: Vec3 = [to.position[0], to.position[1] + to.size[1] / 2, to.position[2]];
    return {
      ...edge,
      arc,
      points: [
        lift,
        [lift[0] + dx * 0.15, top, lift[2]],
        [land[0] - dx * 0.15, top, land[2]],
        land,
      ],
    };
  });

  // Frames around open groups, innermost first so parents can wrap their children.
  const members = new Map<string, LayoutItem[]>();
  for (const item of items) {
    for (const groupId of item.groupPath) {
      (members.get(groupId) ?? members.set(groupId, []).get(groupId)!).push(item);
    }
  }
  const levelOf = new Map<string, number>();
  const level = (groupId: string): number => {
    const cached = levelOf.get(groupId);
    if (cached !== undefined) return cached;
    const children = ir.index.children[groupId]?.groups.filter((id) => members.has(id)) ?? [];
    const value = children.length === 0 ? 0 : 1 + Math.max(...children.map(level));
    levelOf.set(groupId, value);
    return value;
  };
  const frames: LayoutFrame[] = [];
  for (const [groupId, inside] of members) {
    const group = ir.groups[groupId];
    if (!group || group.kind === "model") continue;
    const pad = 0.2 + 0.16 * level(groupId);
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (const item of inside) {
      for (let axis = 0; axis < 3; axis++) {
        min[axis] = Math.min(min[axis]!, item.position[axis]! - item.size[axis]! / 2 - pad);
        max[axis] = Math.max(max[axis]!, item.position[axis]! + item.size[axis]! / 2 + pad);
      }
    }
    min[1] = Math.max(0.02, min[1]);
    frames.push({ groupId, label: group.label, kind: group.kind, level: level(groupId), min, max });
  }

  const min: Vec3 = [0, 0, 0];
  const max: Vec3 = [0, centreY * 2, 0];
  for (const item of items) {
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis]!, item.position[axis]! - item.size[axis]! / 2);
      max[axis] = Math.max(max[axis]!, item.position[axis]! + item.size[axis]! / 2);
    }
  }
  for (const edge of edges) if (edge.arc) max[1] = Math.max(max[1], edge.points[1][1]);
  return { items, edges, frames, bounds: { min, max } };
}

/** A point on a cubic Bézier, t in 0..1. */
export function bezierPoint(points: LayoutEdge["points"], t: number): Vec3 {
  const u = 1 - t;
  const weights = [u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t];
  return [0, 1, 2].map((axis) =>
    points.reduce((sum, point, i) => sum + point[axis]! * weights[i]!, 0),
  ) as Vec3;
}
