/**
 * Lens: from a selection of code to the part of the model it defines.
 *
 * The result is a focus: the layers to isolate and simulate, every other copy
 * of them (a block repeated four times is selected through one definition),
 * and the layers just outside that feed it and read from it.
 */
import type { RuntimeIR } from "../ir/runtime";
import type { CodeRef } from "../ir/wire";

/** A stretch of code: lines of one cell, 1-based and inclusive. */
export type CodeRange = { cell: number; startLine: number; endLine: number };

/** A stretch of code that belongs to a node or a group. */
export type LensInterval = CodeRange & { role: CodeRef["role"]; nodeId?: string; groupId?: string };

export type LensFocus = {
  /**
   * layers: specific layers the selection creates or applies.
   * block: one copy of a repeated block, picked through its definition.
   * group: a whole group (a loop, a stage, a head).
   * symbol: every layer that depends on a selected constant.
   */
  kind: "layers" | "block" | "group" | "symbol";
  /** What to call the focus: a group's label, a layer's label, or a symbol's name. */
  label: string;
  /** The layers that are isolated and simulated, in flow order. */
  nodeIds: string[];
  /** Those layers plus every other copy of them, for highlighting. */
  allNodeIds: string[];
  /** How many copies of the focused block the model has (1 when it is not repeated). */
  copies: number;
  groupId: string | null;
  symbols: string[];
  /** Layers outside the focus whose output it reads. */
  inputs: string[];
  /** Layers outside the focus that read its output. */
  outputs: string[];
};

export type LensQuery = {
  ranges: CodeRange[];
  /** Constants assigned inside the selection (e.g. "PATCH_SIZE"), found by the caller. */
  symbols?: string[];
  /** Code locations to match against. Defaults to the IR's own CodeRefs. */
  intervals?: LensInterval[];
};

const overlaps = (a: CodeRange, b: CodeRange) =>
  a.cell === b.cell && a.startLine <= b.endLine && b.startLine <= a.endLine;
const contains = (outer: CodeRange, inner: CodeRange) =>
  outer.cell === inner.cell && outer.startLine <= inner.startLine && inner.endLine <= outer.endLine;

/** Nodes of a group and of every group inside it, in flow order. */
function nodesInGroup(ir: RuntimeIR, order: string[], groupId: string): string[] {
  return order.filter((id) => {
    for (
      let group: string | null = ir.nodes[id]!.groupId;
      group;
      group = ir.groups[group]?.parentId ?? null
    ) {
      if (group === groupId) return true;
    }
    return false;
  });
}

/** Adds the layers that lie on a path between two selected layers. */
function closeGaps(ir: RuntimeIR, order: string[], selected: Set<string>): string[] {
  const reach = (start: Set<string>, edges: RuntimeIR["index"]["outEdges"], end: "to" | "from") => {
    const seen = new Set(start);
    const queue = [...start];
    while (queue.length > 0) {
      for (const edge of edges[queue.pop()!] ?? []) {
        if (!seen.has(edge[end])) {
          seen.add(edge[end]);
          queue.push(edge[end]);
        }
      }
    }
    return seen;
  };
  const after = reach(selected, ir.index.outEdges, "to");
  const before = reach(selected, ir.index.inEdges, "from");
  return order.filter((id) => after.has(id) && before.has(id));
}

function finish(
  ir: RuntimeIR,
  order: string[],
  focus: Pick<LensFocus, "kind" | "label" | "groupId" | "symbols"> & {
    nodeIds: string[];
    allNodeIds?: string[];
    copies?: number;
  },
): LensFocus | null {
  const inside = new Set(focus.nodeIds);
  const nodeIds = order.filter((id) => inside.has(id));
  if (nodeIds.length === 0) return null;
  const everything = new Set([...nodeIds, ...(focus.allNodeIds ?? [])]);
  const inputs = new Set<string>();
  const outputs = new Set<string>();
  for (const id of nodeIds) {
    for (const edge of ir.index.inEdges[id] ?? [])
      if (!inside.has(edge.from)) inputs.add(edge.from);
    for (const edge of ir.index.outEdges[id] ?? []) if (!inside.has(edge.to)) outputs.add(edge.to);
  }
  return {
    kind: focus.kind,
    label: focus.label,
    nodeIds,
    allNodeIds: order.filter((id) => everything.has(id)),
    copies: focus.copies ?? 1,
    groupId: focus.groupId,
    symbols: focus.symbols,
    inputs: order.filter((id) => inputs.has(id)),
    outputs: order.filter((id) => outputs.has(id)),
  };
}

/** The canonical id shared by every unrolled copy of a node or group. */
const templateOf = (item: { id: string; templateId: string | null }) => item.templateId ?? item.id;

export function resolveSelection(
  ir: RuntimeIR,
  modelId: string,
  query: LensQuery,
): LensFocus | null {
  const order = ir.index.topoOrder[modelId] ?? [];
  const inModel = new Set(order);
  const intervals = (query.intervals ?? ir.index.codeIntervals).filter(
    (interval) =>
      (interval.nodeId !== undefined && inModel.has(interval.nodeId)) ||
      (interval.groupId !== undefined && ir.groups[interval.groupId]?.modelId === modelId),
  );
  const hit = (test: (range: CodeRange, interval: LensInterval) => boolean) =>
    intervals.filter((interval) => query.ranges.some((range) => test(range, interval)));

  // 1. A group whose definition or loop lies entirely inside the selection. The largest one wins.
  const groupHits = hit(contains)
    .filter(
      (interval) => interval.groupId !== undefined && ir.groups[interval.groupId]!.kind !== "model",
    )
    .map((interval) => ({
      interval,
      group: ir.groups[interval.groupId!]!,
      span: interval.endLine - interval.startLine,
    }));
  if (groupHits.length > 0) {
    const widest = Math.max(...groupHits.map((entry) => entry.span));
    const candidates = groupHits.filter((entry) => entry.span === widest);
    // A loop means "all the repetitions"; a definition means "one copy of the block".
    const loop = candidates.find(
      (entry) => entry.interval.role === "loop" && entry.group.kind === "repeat",
    );
    const firstCopy = candidates.find(
      (entry) => entry.group.templateId !== null && entry.group.id.endsWith("#1"),
    );
    if (loop && !candidates.some((entry) => entry.interval.role === "definition")) {
      return finish(ir, order, {
        kind: "group",
        label: loop.group.label,
        groupId: loop.group.id,
        symbols: [],
        nodeIds: nodesInGroup(ir, order, loop.group.id),
        copies: loop.group.repeatCount ?? 1,
      });
    }
    if (firstCopy) {
      const template = templateOf(firstCopy.group);
      const copies = Object.values(ir.groups).filter((group) => group.templateId === template);
      return finish(ir, order, {
        kind: "block",
        label: ir.groups[template]?.label ?? firstCopy.group.label,
        groupId: firstCopy.group.id,
        symbols: [],
        nodeIds: nodesInGroup(ir, order, firstCopy.group.id),
        allNodeIds: copies.flatMap((group) => nodesInGroup(ir, order, group.id)),
        copies: copies.length,
      });
    }
    const group = candidates[0]!.group;
    return finish(ir, order, {
      kind: "group",
      label: group.label,
      groupId: group.id,
      symbols: [],
      nodeIds: nodesInGroup(ir, order, group.id),
    });
  }

  // 2. Layers the selected lines create or apply.
  const nodeHits = [
    ...new Set(hit(overlaps).flatMap((interval) => (interval.nodeId ? [interval.nodeId] : []))),
  ];
  if (nodeHits.length > 0) {
    // Lines inside a repeated block match every copy: isolate the first, highlight them all.
    const repeated = nodeHits.filter((id) => ir.nodes[id]!.templateId !== null);
    if (repeated.length === nodeHits.length) {
      const templates = new Set(repeated.map((id) => templateOf(ir.nodes[id]!)));
      const firstCopy = repeated.filter((id) => id.endsWith("#1"));
      const chosen = firstCopy.length > 0 ? firstCopy : repeated.slice(0, templates.size);
      const nodeIds = closeGaps(ir, order, new Set(chosen));
      return finish(ir, order, {
        kind: "layers",
        label: nodeIds.length === 1 ? ir.nodes[nodeIds[0]!]!.label : `${nodeIds.length} layers`,
        groupId: null,
        symbols: [],
        nodeIds,
        allNodeIds: repeated,
        copies: Math.max(1, Math.round(repeated.length / templates.size)),
      });
    }
    const nodeIds = closeGaps(ir, order, new Set(nodeHits));
    return finish(ir, order, {
      kind: "layers",
      label: nodeIds.length === 1 ? ir.nodes[nodeIds[0]!]!.label : `${nodeIds.length} layers`,
      groupId: null,
      symbols: [],
      nodeIds,
    });
  }

  // 3. A constant: every layer whose settings were written with it, and every block it repeats.
  const symbols = (query.symbols ?? []).filter((name) =>
    ir.symbols.some((symbol) => symbol.name === name),
  );
  if (symbols.length > 0) {
    const using = new Set(
      order.filter((id) => ir.nodes[id]!.symbolsUsed.some((name) => symbols.includes(name))),
    );
    for (const group of Object.values(ir.groups)) {
      if (group.modelId === modelId && group.repeatSymbol && symbols.includes(group.repeatSymbol)) {
        for (const id of nodesInGroup(ir, order, group.id)) using.add(id);
      }
    }
    return finish(ir, order, {
      kind: "symbol",
      label: symbols.join(", "),
      groupId: null,
      symbols,
      nodeIds: [...using],
    });
  }
  return null;
}

/** A sentence for the focus banner. */
export function describeFocus(focus: LensFocus): string {
  const layers = `${focus.nodeIds.length} layer${focus.nodeIds.length === 1 ? "" : "s"}`;
  if (focus.kind === "symbol") {
    return `${layers} depend${focus.nodeIds.length === 1 ? "s" : ""} on ${focus.label}`;
  }
  if (focus.kind === "block") return `${focus.label} · one of ${focus.copies} copies · ${layers}`;
  if (focus.copies > 1 && focus.kind === "group")
    return `${focus.label} · all ${focus.copies} repetitions · ${layers}`;
  if (focus.copies > 1) return `${focus.label} · in each of ${focus.copies} copies`;
  return focus.kind === "group" ? `${focus.label} · ${layers}` : focus.label;
}
