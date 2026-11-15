/**
 * The 2D graph layout: the same visible graph as the 3D view, drawn top to
 * bottom with fixed-size cards. Ranks become rows and lanes become columns.
 */
import type { RuntimeIR } from "../ir/runtime";
import { buildVisibleGraph, type VisibleEdge, type VisibleItem } from "./graph";

export const CARD_WIDTH = 220;
export const CARD_HEIGHT = 58;
const ROW_GAP = 34;
const COLUMN_GAP = 56;
const FRAME_PAD = 14;
const FRAME_LABEL = 20;

export type GraphCard = VisibleItem & { x: number; y: number; width: number; height: number };

export type GraphFrame = {
  groupId: string;
  label: string;
  level: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type GraphLayout = { cards: GraphCard[]; edges: VisibleEdge[]; frames: GraphFrame[] };

export function layoutModel2D(
  ir: RuntimeIR,
  modelId: string,
  collapsed: ReadonlySet<string>,
): GraphLayout {
  const graph = buildVisibleGraph(ir, modelId, collapsed);

  // Rows that start or end a framed group get extra room for the frame's border and label.
  const nesting = (item: VisibleItem) =>
    item.groupPath.filter((id) => ir.groups[id]?.kind !== "model").length;
  const maxNesting = graph.items.reduce((max, item) => Math.max(max, nesting(item)), 0);
  const rowStep = CARD_HEIGHT + ROW_GAP + (maxNesting > 0 ? FRAME_LABEL : 0);
  const columnStep = CARD_WIDTH + COLUMN_GAP + maxNesting * FRAME_PAD * 2;

  const cards: GraphCard[] = graph.items.map((item) => ({
    ...item,
    x: item.lane * columnStep,
    y: item.rank * rowStep,
    width: CARD_WIDTH,
    height: CARD_HEIGHT,
  }));

  const members = new Map<string, GraphCard[]>();
  for (const card of cards) {
    for (const groupId of card.groupPath) {
      if (ir.groups[groupId]?.kind === "model") continue;
      (members.get(groupId) ?? members.set(groupId, []).get(groupId)!).push(card);
    }
  }
  const levels = new Map<string, number>();
  const level = (groupId: string): number => {
    const cached = levels.get(groupId);
    if (cached !== undefined) return cached;
    const children = ir.index.children[groupId]?.groups.filter((id) => members.has(id)) ?? [];
    const value = children.length === 0 ? 0 : 1 + Math.max(...children.map(level));
    levels.set(groupId, value);
    return value;
  };

  const frames: GraphFrame[] = [...members].map(([groupId, inside]) => {
    const pad = FRAME_PAD * (level(groupId) + 1);
    const left = Math.min(...inside.map((card) => card.x)) - pad;
    const top = Math.min(...inside.map((card) => card.y)) - pad - FRAME_LABEL;
    const right = Math.max(...inside.map((card) => card.x + card.width)) + pad;
    const bottom = Math.max(...inside.map((card) => card.y + card.height)) + pad;
    return {
      groupId,
      label: ir.groups[groupId]!.label,
      level: level(groupId),
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
    };
  });
  // Outer frames first, so inner ones are drawn on top of them.
  frames.sort((a, b) => b.level - a.level);

  return { cards, edges: graph.edges, frames };
}
