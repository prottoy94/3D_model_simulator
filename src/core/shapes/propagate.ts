/**
 * The shape engine (doc 03 §4): walks each model in topological order and
 * computes every node's output shape, parameter count and FLOPs from the op
 * catalog. Golden rule 2: these numbers are the ones the UI shows.
 */
import type { NormalizedIR } from "../ir/normalize";
import type { Issue, RuntimeNode } from "../ir/runtime";
import type { WireEdge } from "../ir/wire";
import { getOp } from "../ops";
import type { AnyOpSpec, ShapeCtx } from "../ops/types";
import {
  BATCH,
  isKnown,
  layoutForRank,
  ShapeError,
  shapesEqual,
  shapeText,
  type Dim,
  type Framework,
  type TensorShape,
} from "./tensor";

export type PropagationResult = {
  topoOrder: Record<string, string[]>;
  inEdges: Record<string, WireEdge[]>;
  outEdges: Record<string, WireEdge[]>;
  llmDisagreements: Array<{ nodeId: string; claimed: string; computed: string }>;
};

export function claimedShape(
  claimed: Array<number | string> | null,
  framework: Framework,
): TensorShape | null {
  if (!claimed || claimed.length === 0) return null;
  const dims = claimed.map((dim): Dim =>
    typeof dim === "number" ? dim : { sym: dim === "None" ? "N" : dim },
  );
  if (isKnown(dims[0]) === false && (dims[0] as { sym: string }).sym === "N") dims[0] = BATCH;
  return { dims, layout: layoutForRank(dims.length, framework) };
}

/** Removes axis 1 (the per-item axis of a TimeDistributed wrapper). */
function foldTime(shape: TensorShape, framework: Framework): { shape: TensorShape; time: Dim } {
  if (shape.dims.length < 3) {
    throw new ShapeError(
      `TimeDistributed needs at least (B, items, size), got ${shapeText(shape)}`,
    );
  }
  const dims = [shape.dims[0]!, ...shape.dims.slice(2)];
  return { shape: { dims, layout: layoutForRank(dims.length, framework) }, time: shape.dims[1]! };
}

function compute(node: RuntimeNode, spec: AnyOpSpec, inputs: TensorShape[], ctx: ShapeCtx) {
  if (node.wrapper !== "time_distributed") {
    const out = spec.inferShape(inputs, node.params, ctx);
    return {
      out,
      count: spec.countParams(inputs, node.params, ctx),
      flops: spec.flops(inputs, out, node.params),
    };
  }
  const folded = inputs.map((shape) => foldTime(shape, ctx.framework));
  const inner = folded.map((entry) => entry.shape);
  const time = folded[0]?.time ?? { sym: "T" };
  const innerOut = spec.inferShape(inner, node.params, ctx);
  const dims = [innerOut.dims[0] ?? BATCH, time, ...innerOut.dims.slice(1)];
  return {
    out: { dims, layout: dims.length === 3 ? ("NLC" as const) : ("any" as const) },
    count: spec.countParams(inner, node.params, ctx),
    // Per item when the item count is only known at run time.
    flops: spec.flops(inner, innerOut, node.params) * (isKnown(time) ? time : 1),
  };
}

export function propagateShapes(ir: NormalizedIR): PropagationResult {
  const ctx: ShapeCtx = { framework: ir.framework };
  const nodeById = new Map(ir.nodes.map((node) => [node.id, node]));
  const modelOfGroup = new Map(ir.groups.map((group) => [group.id, group.modelId]));

  const inEdges: Record<string, WireEdge[]> = {};
  const outEdges: Record<string, WireEdge[]> = {};
  for (const node of ir.nodes) {
    inEdges[node.id] = [];
    outEdges[node.id] = [];
  }
  for (const edge of ir.edges) {
    inEdges[edge.to]?.push(edge);
    outEdges[edge.from]?.push(edge);
  }
  for (const edges of Object.values(inEdges)) edges.sort((a, b) => a.toPort - b.toPort);

  // Kahn's algorithm, keeping the IR's own node order among ready nodes.
  const remaining = new Map(ir.nodes.map((node) => [node.id, inEdges[node.id]!.length]));
  const order: string[] = [];
  const ready = ir.nodes.filter((node) => remaining.get(node.id) === 0).map((node) => node.id);
  const position = new Map(ir.nodes.map((node, index) => [node.id, index]));
  while (ready.length > 0) {
    ready.sort((a, b) => position.get(a)! - position.get(b)!);
    const id = ready.shift()!;
    order.push(id);
    for (const edge of outEdges[id]!) {
      const left = remaining.get(edge.to)! - 1;
      remaining.set(edge.to, left);
      if (left === 0) ready.push(edge.to);
    }
  }
  if (order.length < ir.nodes.length) {
    ir.issues.push({
      code: "cycle",
      message: "The graph has a loop, so some layers could not be ordered or computed.",
      severity: "error",
    });
  }

  const tainted = new Set<string>();
  const llmDisagreements: PropagationResult["llmDisagreements"] = [];
  const ordered = new Set(order);

  for (const node of ir.nodes) {
    if (!ordered.has(node.id)) {
      node.status = "error";
      node.issues.push({
        code: "in-cycle",
        message: "This layer is part of a loop in the graph.",
        severity: "error",
      });
    }
  }

  for (const id of order) {
    const node = nodeById.get(id)!;
    const claimed = claimedShape(node.claimedOutputShape, ir.framework);
    const sources = inEdges[id]!.map((edge) => nodeById.get(edge.from)!);
    const missing = sources.some((source) => source.outShape === null);
    const upstreamTainted = sources.some((source) => tainted.has(source.id));
    node.inShapes = sources.flatMap((source) => (source.outShape ? [source.outShape] : []));

    const fail = (issue: Issue, status: RuntimeNode["status"]) => {
      node.issues.push(issue);
      node.status = status;
      node.outShape = claimed;
      tainted.add(id);
    };

    const spec = getOp(node.op);
    if (node.status === "error") {
      // Bad params or wiring were already reported by normalization.
      node.outShape = claimed;
      tainted.add(id);
      continue;
    }
    if (!spec) {
      node.outShape = claimed;
      node.status = "unverified";
      tainted.add(id);
      continue;
    }
    if (missing) {
      fail(
        {
          code: "input-unknown",
          message: "An input shape is unknown because an earlier layer failed.",
          severity: "warn",
        },
        "unverified",
      );
      continue;
    }

    try {
      const result = compute(node, spec, node.inShapes, ctx);
      node.outShape = result.out;
      node.flops = result.flops;
      node.paramCount =
        node.trainable === "no"
          ? { trainable: 0, nonTrainable: result.count.trainable + result.count.nonTrainable }
          : result.count;
      if (node.trainable === "partial") {
        node.issues.push({
          code: "partly-frozen",
          message:
            "Only part of this layer is trainable. All of its weights are counted as trainable here.",
          severity: "info",
        });
      }
      if (spec.declared) {
        node.status = "unverified";
        node.issues.push({
          code: "declared",
          message: "The shape and parameter count of this layer are declared, not computed.",
          severity: "info",
        });
      } else if (upstreamTainted) {
        node.status = "unverified";
        tainted.add(id);
      } else {
        node.status = "ok";
      }
      if (claimed && !spec.declared && !shapesEqual(claimed, result.out)) {
        llmDisagreements.push({
          nodeId: id,
          claimed: shapeText(claimed),
          computed: shapeText(result.out),
        });
        node.issues.push({
          code: "claimed-shape-differs",
          message: `The IR claimed ${shapeText(claimed)} but the engine computed ${shapeText(result.out)}. The computed shape is used.`,
          severity: "warn",
        });
      }
    } catch (error) {
      if (!(error instanceof ShapeError)) throw error;
      fail({ code: "shape-error", message: error.message, severity: "error" }, "error");
    }
  }

  const topoOrder: Record<string, string[]> = {};
  for (const model of ir.models) topoOrder[model.id] = [];
  for (const id of order) {
    const modelId = modelOfGroup.get(nodeById.get(id)!.groupId);
    if (modelId !== undefined) (topoOrder[modelId] ??= []).push(id);
  }
  return { topoOrder, inEdges, outEdges, llmDisagreements };
}
