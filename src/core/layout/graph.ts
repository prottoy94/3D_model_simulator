/**
 * The visible graph: what is drawn once collapsed groups are folded into a
 * single item. Both the 3D and the 2D layout start from this, so the two views
 * always show the same structure.
 */
import type { NodeStatus, RuntimeIR } from "../ir/runtime";
import type { WireEdge } from "../ir/wire";
import { getOp } from "../ops";
import type { OpFamily, OpKind } from "../ops/kinds";
import type { GlyphKind } from "../ops/types";
import type { TensorShape } from "../shapes/tensor";

export type VisibleItem = {
  /** A node id, or the id of a collapsed group. */
  id: string;
  kind: "node" | "group";
  op: OpKind | null;
  glyph: GlyphKind;
  family: OpFamily;
  label: string;
  outShape: TensorShape | null;
  status: NodeStatus;
  /** The nodes this item stands for (one, or every node of a collapsed group). */
  memberIds: string[];
  params: number;
  /** Groups this item sits in, innermost first. Collapsed groups are not listed. */
  groupPath: string[];
  /** Column along the flow: 0 for inputs, then longest path from an input. */
  rank: number;
  /** Row across the flow: 0 is the main path, branches go to ±1, ±2, … */
  lane: number;
};

export type VisibleEdge = {
  id: string;
  from: string;
  to: string;
  kind: WireEdge["kind"];
  /** How many columns the edge crosses. Skips are drawn as arcs. */
  span: number;
};

export type VisibleGraph = {
  items: VisibleItem[];
  edges: VisibleEdge[];
  ranks: number;
  lanes: [number, number];
};

const STATUS_ORDER: NodeStatus[] = ["ok", "unverified", "mismatch", "error"];
const worst = (a: NodeStatus, b: NodeStatus) =>
  STATUS_ORDER.indexOf(b) > STATUS_ORDER.indexOf(a) ? b : a;

/** Groups that start collapsed: the ones the IR marks as too big to show open. */
export function defaultCollapsed(ir: RuntimeIR): string[] {
  return Object.values(ir.groups)
    .filter((group) => !group.expandedByDefault && group.kind !== "model")
    .map((group) => group.id);
}

export function buildVisibleGraph(
  ir: RuntimeIR,
  modelId: string,
  collapsed: ReadonlySet<string>,
): VisibleGraph {
  const order = ir.index.topoOrder[modelId] ?? [];
  const items = new Map<string, VisibleItem>();
  const itemOfNode = new Map<string, string>();

  for (const nodeId of order) {
    const node = ir.nodes[nodeId]!;
    // Walk up the group chain; the outermost collapsed group swallows the node.
    const chain: string[] = [];
    let holder: string | null = null;
    for (let id: string | null = node.groupId; id && ir.groups[id]; id = ir.groups[id]!.parentId) {
      if (collapsed.has(id) && ir.groups[id]!.kind !== "model") {
        holder = id;
        chain.length = 0;
      } else chain.push(id);
    }
    const count = node.paramCount.trainable + node.paramCount.nonTrainable;

    if (holder === null) {
      const spec = getOp(node.op);
      items.set(nodeId, {
        id: nodeId,
        kind: "node",
        op: node.op,
        glyph: spec?.glyph ?? "box",
        family: spec?.family ?? "custom",
        label: node.label,
        outShape: node.outShape,
        status: node.status,
        memberIds: [nodeId],
        params: count,
        groupPath: chain,
        rank: 0,
        lane: 0,
      });
      itemOfNode.set(nodeId, nodeId);
      continue;
    }

    const group = ir.groups[holder]!;
    const existing = items.get(holder);
    if (existing) {
      existing.memberIds.push(nodeId);
      existing.params += count;
      existing.status = worst(existing.status, node.status);
      existing.outShape = node.outShape ?? existing.outShape; // the last member's output
    } else {
      items.set(holder, {
        id: holder,
        kind: "group",
        op: null,
        glyph: "capsule",
        family: "composite",
        label: group.label,
        outShape: node.outShape,
        status: node.status,
        memberIds: [nodeId],
        params: count,
        groupPath: chain,
        rank: 0,
        lane: 0,
      });
    }
    itemOfNode.set(nodeId, holder);
  }

  const edges: VisibleEdge[] = [];
  const seen = new Set<string>();
  for (const edge of ir.edges) {
    const from = itemOfNode.get(edge.from);
    const to = itemOfNode.get(edge.to);
    if (!from || !to || from === to) continue;
    const key = `${from}→${to}→${edge.kind}`;
    if (seen.has(key)) continue;
    seen.add(key);
    edges.push({ id: edge.id, from, to, kind: edge.kind, span: 1 });
  }

  // Longest-path ranks. `items` is in topological order because `order` is.
  const incoming = new Map<string, VisibleEdge[]>();
  const outgoing = new Map<string, VisibleEdge[]>();
  for (const edge of edges) {
    (incoming.get(edge.to) ?? incoming.set(edge.to, []).get(edge.to)!).push(edge);
    (outgoing.get(edge.from) ?? outgoing.set(edge.from, []).get(edge.from)!).push(edge);
  }
  for (const item of items.values()) {
    for (const edge of incoming.get(item.id) ?? []) {
      item.rank = Math.max(item.rank, (items.get(edge.from)?.rank ?? 0) + 1);
    }
  }
  for (const edge of edges) edge.span = items.get(edge.to)!.rank - items.get(edge.from)!.rank;

  // Lanes: a node continues its main input's lane; extra children fan out to 1, -1, 2, -2, …
  const taken = new Set<string>();
  const fan = (index: number) => (index === 0 ? 0 : index % 2 === 1 ? (index + 1) / 2 : -index / 2);
  const claim = (rank: number, wanted: number): number => {
    for (let step = 0; ; step++) {
      for (const lane of step === 0 ? [wanted] : [wanted + step, wanted - step]) {
        if (!taken.has(`${rank}:${lane}`)) {
          taken.add(`${rank}:${lane}`);
          return lane;
        }
      }
    }
  };
  let roots = 0;
  for (const item of items.values()) {
    const inputs = incoming.get(item.id) ?? [];
    // The main input is the closest one upstream (a skip connection is never the main path).
    const main = [...inputs].sort((a, b) => a.span - b.span)[0];
    if (!main) {
      item.lane = claim(item.rank, fan(roots++));
      continue;
    }
    const parent = items.get(main.from)!;
    const siblings = (outgoing.get(parent.id) ?? [])
      .filter((edge) => edge.span === 1)
      .map((edge) => edge.to)
      .filter((id, index, all) => all.indexOf(id) === index);
    const index = Math.max(0, siblings.indexOf(item.id));
    // A merge that takes two same-distance inputs sits between them, on the lower lane.
    const peers = inputs
      .filter((edge) => edge.span === main.span)
      .map((edge) => items.get(edge.from)!.lane);
    const base =
      peers.length > 1 ? peers.sort((a, b) => Math.abs(a) - Math.abs(b))[0]! : parent.lane;
    item.lane = claim(item.rank, base + fan(index));
  }

  const all = [...items.values()];
  const laneValues = all.map((item) => item.lane);
  return {
    items: all,
    edges,
    ranks: all.reduce((max, item) => Math.max(max, item.rank), 0) + 1,
    lanes: [Math.min(0, ...laneValues), Math.max(0, ...laneValues)],
  };
}
