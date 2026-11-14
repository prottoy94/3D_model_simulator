/**
 * Wire IR → Runtime IR: normalize, run the shape engine, verify against saved
 * summaries and build the lookup indexes. Zoo expansion (doc 03 §5) slots in
 * between normalization and the shape engine in Phase 4.
 */
import type { ModelSummary } from "../notebook/summary";
import { propagateShapes } from "../shapes/propagate";
import { sumParams, verifyModel } from "../shapes/verify";
import { normalizeWire } from "./normalize";
import type {
  BuildOptions,
  CodeInterval,
  RuntimeGroup,
  RuntimeIR,
  RuntimeNode,
  VerificationReport,
} from "./runtime";

export type BuildResult = { ok: true; ir: RuntimeIR } | { ok: false; errors: string[] };

const plain = (text: string) => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** Pairs each model with the saved summary that carries its name, else by position among the unnamed ones. */
function pickSummary(
  modelName: string,
  summaries: ModelSummary[],
  taken: Set<ModelSummary>,
): ModelSummary | null {
  const byName = summaries.find(
    (summary) =>
      !taken.has(summary) &&
      summary.modelName !== null &&
      plain(summary.modelName) === plain(modelName),
  );
  if (byName) taken.add(byName);
  return byName ?? null;
}

export function buildRuntimeIR(input: unknown, options: BuildOptions = {}): BuildResult {
  const normalized = normalizeWire(input, options.provenance);
  if (!normalized.ok) return normalized;
  const ir = normalized.ir;
  const { topoOrder, inEdges, outEdges, llmDisagreements } = propagateShapes(ir);

  const nodes: Record<string, RuntimeNode> = Object.fromEntries(
    ir.nodes.map((node) => [node.id, node]),
  );
  const groups: Record<string, RuntimeGroup> = Object.fromEntries(
    ir.groups.map((group) => [group.id, group]),
  );

  const children: RuntimeIR["index"]["children"] = {};
  for (const group of ir.groups) children[group.id] = { groups: [], nodes: [] };
  for (const group of ir.groups)
    if (group.parentId) children[group.parentId]?.groups.push(group.id);
  for (const node of ir.nodes) children[node.groupId]?.nodes.push(node.id);

  const verification: VerificationReport = { perModel: {}, llmDisagreements, repairAttempts: 0 };
  const taken = new Set<ModelSummary>();
  for (const model of ir.models) {
    const modelNodes = (topoOrder[model.id] ?? []).map((id) => nodes[id]!);
    const modelGroups = ir.groups.filter((group) => group.modelId === model.id);
    const summary = pickSummary(model.name, options.summaries ?? [], taken);
    verification.perModel[model.id] = verifyModel({
      nodes: modelNodes,
      groups: modelGroups,
      summary,
    });
  }

  // Group stats, deepest groups first so parents can add up their children.
  const depthOf = (group: RuntimeGroup): number => {
    let depth = 0;
    for (let current = group; current.parentId && groups[current.parentId] && depth < 64; depth++) {
      current = groups[current.parentId]!;
    }
    return depth;
  };
  for (const group of ir.groups) group.stats.depth = depthOf(group);
  for (const group of [...ir.groups].sort((a, b) => b.stats.depth - a.stats.depth)) {
    const own = children[group.id]!.nodes.map((id) => nodes[id]!);
    const totals = sumParams(own);
    let flops = own.reduce((sum, node) => sum + node.flops, 0);
    let nodeCount = own.length;
    let { trainable, nonTrainable } = totals;
    for (const childId of children[group.id]!.groups) {
      const child = groups[childId]!.stats;
      trainable += child.params.trainable;
      nonTrainable += child.params.nonTrainable;
      flops += child.flops;
      nodeCount += child.nodeCount;
    }
    group.stats = {
      params: { trainable, nonTrainable },
      flops,
      nodeCount,
      depth: group.stats.depth,
    };
    // Big backbones and long repeats start collapsed so the first view stays readable.
    group.expandedByDefault = group.kind !== "backbone" && nodeCount <= 60;
  }

  const codeIntervals: CodeInterval[] = [];
  for (const node of ir.nodes)
    for (const ref of node.refs) codeIntervals.push({ ...ref, nodeId: node.id });
  for (const group of ir.groups)
    for (const ref of group.refs) codeIntervals.push({ ...ref, groupId: group.id });
  codeIntervals.sort(
    (a, b) => a.cell - b.cell || a.startLine - b.startLine || a.endLine - b.endLine,
  );

  const { nodes: _wireNodes, groups: _wireGroups, ...rest } = ir.wire;
  void _wireNodes;
  void _wireGroups;

  return {
    ok: true,
    ir: {
      ...rest,
      models: ir.models,
      edges: ir.edges,
      engineFramework: ir.framework,
      nodes,
      groups,
      index: { topoOrder, children, inEdges, outEdges, codeIntervals },
      issues: ir.issues,
      verification,
      assumptions: ir.wire.symbols
        .filter((symbol) => symbol.provenance === "assumed" || symbol.provenance === "from-output")
        .map((symbol) => ({
          symbol: symbol.name,
          value: symbol.value,
          source: symbol.note || symbol.provenance,
          editable: true as const,
        })),
    },
  };
}

/** Totals for one model, as the UI shows them. */
export function modelTotals(ir: RuntimeIR, modelId: string) {
  return ir.verification.perModel[modelId]?.totals ?? null;
}
