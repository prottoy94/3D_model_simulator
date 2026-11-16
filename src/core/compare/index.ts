/** Side-by-side facts about two models, all computed from the Runtime IR. */
import type { RuntimeIR } from "../ir";
import { getOp, type OpFamily } from "../ops";
import { shapeText } from "../shapes/tensor";

export type FamilyShare = { family: OpFamily; layers: number; params: number };

export type ModelStats = {
  name: string;
  framework: string;
  layers: number;
  /** Layers on the longest path from an input to an output. */
  depth: number;
  params: number;
  trainable: number;
  nonTrainable: number;
  /** Multiply-adds × 2 for one sample. */
  flops: number;
  inputs: string[];
  outputs: string[];
  /** Families that occur in the model, largest parameter share first. */
  families: FamilyShare[];
  /** The layer that holds the most parameters. */
  heaviest: { label: string; params: number } | null;
  /** False when some layer could not be computed, so the totals are a lower bound. */
  complete: boolean;
};

export function modelStats(ir: RuntimeIR, modelId: string): ModelStats | null {
  const model = ir.models.find((candidate) => candidate.id === modelId);
  const totals = ir.verification.perModel[modelId]?.totals.computed;
  if (!model || !totals) return null;
  const order = ir.index.topoOrder[modelId] ?? [];

  const depthOf: Record<string, number> = {};
  const families = new Map<OpFamily, FamilyShare>();
  let flops = 0;
  let heaviest: ModelStats["heaviest"] = null;
  let complete = true;

  for (const id of order) {
    const node = ir.nodes[id];
    if (!node) continue;
    const before = (ir.index.inEdges[id] ?? []).map((edge) => depthOf[edge.from] ?? 0);
    depthOf[id] = 1 + Math.max(0, ...before);
    flops += node.flops;
    if (node.status === "error" || node.status === "unverified") complete = false;

    const family = getOp(node.op)?.family ?? "custom";
    const share = families.get(family) ?? { family, layers: 0, params: 0 };
    share.layers += 1;
    // A node that reuses another node's weights adds no parameters of its own.
    const params = node.sharedWith ? 0 : node.paramCount.trainable + node.paramCount.nonTrainable;
    share.params += params;
    families.set(family, share);
    if (params > 0 && (!heaviest || params > heaviest.params)) {
      heaviest = { label: node.label, params };
    }
  }

  const shapes = (ids: string[]) =>
    ids.map((id) => {
      const shape = ir.nodes[id]?.outShape;
      return shape ? shapeText(shape) : "unknown";
    });

  return {
    name: model.name,
    framework: ir.engineFramework,
    layers: order.length,
    depth: Math.max(0, ...Object.values(depthOf)),
    params: totals.total,
    trainable: totals.trainable,
    nonTrainable: totals.nonTrainable,
    flops,
    inputs: shapes(model.inputNodeIds),
    outputs: shapes(model.outputNodeIds),
    families: [...families.values()].sort((a, b) => b.params - a.params || b.layers - a.layers),
    heaviest,
    complete,
  };
}

/** "3.1× larger", "12% smaller" or "the same": how `a` relates to `b`. */
export function ratioText(a: number, b: number, more = "more", less = "fewer"): string {
  if (a === b) return "the same";
  if (a === 0 || b === 0) return a > b ? more : less;
  const big = Math.max(a, b) / Math.min(a, b);
  const amount =
    big >= 2
      ? `${big >= 10 ? Math.round(big) : big.toFixed(1)}×`
      : `${Math.round((big - 1) * 100)}%`;
  return `${amount} ${a > b ? more : less}`;
}
