/**
 * Normalization, steps 1–5 of doc 03 §2: validate the wire IR, substitute
 * symbols, type the params through the op catalog, unroll repeat groups and
 * check graph integrity. Shapes are added afterwards by the shape engine.
 */
import { getOp } from "../ops";
import type { Framework } from "../shapes/tensor";
import type { Issue, RuntimeGroup, RuntimeNode } from "./runtime";
import {
  TIME_DISTRIBUTED_KEY,
  wireIRSchema,
  type ParamEntry,
  type WireEdge,
  type WireIR,
  type WireModel,
} from "./wire";

export type NormalizedIR = {
  wire: WireIR;
  framework: Framework;
  symbols: Map<string, unknown>;
  models: WireModel[];
  nodes: RuntimeNode[];
  groups: RuntimeGroup[];
  edges: WireEdge[];
  issues: Issue[];
};

export type NormalizeResult = { ok: true; ir: NormalizedIR } | { ok: false; errors: string[] };

export function engineFramework(framework: WireIR["framework"]): Framework {
  return framework === "pytorch" || framework === "ultralytics" || framework === "huggingface"
    ? "pytorch"
    : "keras";
}

export function normalizeWire(
  input: unknown,
  provenance: RuntimeNode["provenance"] = "ai",
): NormalizeResult {
  const parsed = wireIRSchema.safeParse(input);
  if (!parsed.success) {
    return {
      ok: false,
      errors: parsed.error.issues.map(
        (issue) => `${issue.path.join(".") || "(root)"}: ${issue.message}`,
      ),
    };
  }
  const wire = parsed.data;
  const framework = engineFramework(wire.framework);
  const issues: Issue[] = [];

  const symbols = new Map<string, unknown>();
  for (const symbol of wire.symbols) symbols.set(symbol.name, parseSymbolValue(symbol.value));

  const nodes = wire.nodes.map((node): RuntimeNode => {
    const {
      params,
      wrapper,
      issues: paramIssues,
      failed,
      symbolsUsed,
    } = typeParams(node.op, node.params, symbols, framework);
    return {
      ...node,
      params,
      wrapper,
      inShapes: [],
      outShape: null,
      paramCount: { trainable: 0, nonTrainable: 0 },
      flops: 0,
      status: failed ? "error" : "unverified",
      issues: paramIssues,
      provenance,
      templateId: null,
      summaryRow: null,
      symbolsUsed,
    };
  });

  const groups = wire.groups.map((group): RuntimeGroup => {
    const fromSymbol = group.repeatSymbol ? symbols.get(group.repeatSymbol) : undefined;
    return {
      ...group,
      repeatCount: group.repeatCount ?? (typeof fromSymbol === "number" ? fromSymbol : null),
      templateId: null,
      stats: { params: { trainable: 0, nonTrainable: 0 }, flops: 0, nodeCount: 0, depth: 0 },
      expandedByDefault: true,
    };
  });

  const graph = {
    nodes,
    groups,
    edges: [...wire.edges],
    models: wire.models.map((model) => ({ ...model })),
  };
  checkUniqueIds(graph, issues);
  unrollRepeats(graph, issues);
  checkIntegrity(graph, issues);

  return { ok: true, ir: { wire, framework, symbols, ...graph, issues } };
}

function parseSymbolValue(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function typeParams(
  op: RuntimeNode["op"],
  entries: ParamEntry[],
  symbols: Map<string, unknown>,
  framework: Framework,
): {
  params: Record<string, unknown>;
  wrapper: RuntimeNode["wrapper"];
  symbolsUsed: string[];
  issues: Issue[];
  failed: boolean;
} {
  const issues: Issue[] = [];
  const symbolsUsed: string[] = [];
  const given: Record<string, unknown> = {};
  for (const { key, value } of entries) {
    // A string that names a symbol stands for that symbol's value ("PATCH_SIZE" → 16).
    const named = typeof value === "string" && symbols.has(value);
    if (named) symbolsUsed.push(value);
    given[key] = named ? symbols.get(value) : value;
  }
  const wrapper = given[TIME_DISTRIBUTED_KEY] === true ? "time_distributed" : null;
  delete given[TIME_DISTRIBUTED_KEY];

  const spec = getOp(op);
  if (!spec) {
    issues.push({
      code: "op-not-in-catalog",
      message: `"${op}" has no catalog entry yet, so its shape and parameter count are not computed.`,
      severity: "warn",
    });
    return { params: given, wrapper, issues, failed: false, symbolsUsed };
  }

  const result = spec.params.safeParse({ ...spec.defaults[framework], ...given });
  if (!result.success) {
    for (const issue of result.error.issues) {
      issues.push({
        code: "bad-param",
        message: `${issue.path.join(".") || "params"}: ${issue.message}`,
        severity: "error",
      });
    }
    return { params: given, wrapper, issues, failed: true, symbolsUsed };
  }
  const params = result.data as Record<string, unknown>;
  for (const key of Object.keys(given)) {
    if (!(key in params)) {
      issues.push({
        code: "unknown-param",
        message: `"${key}" is not a parameter of ${op} and was ignored.`,
        severity: "info",
      });
    }
  }
  return { params, wrapper, issues, failed: false, symbolsUsed };
}

type Graph = {
  nodes: RuntimeNode[];
  groups: RuntimeGroup[];
  edges: WireEdge[];
  models: WireModel[];
};

function checkUniqueIds(graph: Graph, issues: Issue[]): void {
  const seen = new Set<string>();
  const all = [...graph.models, ...graph.groups, ...graph.nodes, ...graph.edges];
  for (const { id } of all) {
    if (seen.has(id))
      issues.push({
        code: "duplicate-id",
        message: `The id "${id}" is used more than once.`,
        severity: "error",
      });
    seen.add(id);
  }
}

/** Groups below `rootId`, not including it. */
function descendantGroups(groups: RuntimeGroup[], rootId: string): RuntimeGroup[] {
  const result: RuntimeGroup[] = [];
  const queue = [rootId];
  while (queue.length > 0) {
    const parent = queue.shift()!;
    for (const group of groups) {
      if (group.parentId === parent && !result.includes(group)) {
        result.push(group);
        queue.push(group.id);
      }
    }
  }
  return result;
}

function groupDepth(groups: RuntimeGroup[], group: RuntimeGroup): number {
  let depth = 0;
  let current: RuntimeGroup | undefined = group;
  const seen = new Set<string>();
  while (current?.parentId && !seen.has(current.id)) {
    seen.add(current.id);
    const parentId: string = current.parentId;
    current = groups.find((candidate) => candidate.id === parentId);
    depth++;
  }
  return depth;
}

/**
 * A repeat group holds one canonical copy of its block. This replaces it with
 * N copies chained one after another (ids get "#1".."#N"), each in its own
 * child group, and remembers the canonical id as `templateId` so Lens can map
 * the block's code to every copy.
 */
function unrollRepeats(graph: Graph, issues: Issue[]): void {
  const repeats = graph.groups
    .filter((group) => group.kind === "repeat")
    .sort((a, b) => groupDepth(graph.groups, b) - groupDepth(graph.groups, a));

  for (const repeat of repeats) {
    const count = repeat.repeatCount;
    if (count === null || !Number.isInteger(count) || count < 1) {
      issues.push({
        code: "repeat-count-unknown",
        message: `"${repeat.label}" repeats an unknown number of times, so one copy is shown.`,
        severity: "warn",
      });
      continue;
    }

    const inner = descendantGroups(graph.groups, repeat.id);
    const groupIds = new Set([repeat.id, ...inner.map((group) => group.id)]);
    const members = graph.nodes.filter((node) => groupIds.has(node.groupId));
    const memberIds = new Set(members.map((node) => node.id));
    if (members.length === 0) continue;

    const internal = graph.edges.filter(
      (edge) => memberIds.has(edge.from) && memberIds.has(edge.to),
    );
    const entering = graph.edges.filter(
      (edge) => !memberIds.has(edge.from) && memberIds.has(edge.to),
    );
    const leaving = graph.edges.filter(
      (edge) => memberIds.has(edge.from) && !memberIds.has(edge.to),
    );

    // The node whose output is handed to the next copy.
    let exits = [...new Set(leaving.map((edge) => edge.from))];
    if (exits.length === 0) {
      exits = members
        .filter((node) => !internal.some((edge) => edge.from === node.id))
        .map((node) => node.id);
    }
    const exit = exits.length === 1 ? exits[0]! : null;
    if (!exit && count > 1) {
      issues.push({
        code: "repeat-exit-ambiguous",
        message: `"${repeat.label}" has ${exits.length} outputs, so its copies could not be chained.`,
        severity: "warn",
      });
    }

    // The outside tensor that flows through the block (the most common entry source).
    const sourceCounts = new Map<string, number>();
    for (const edge of entering)
      sourceCounts.set(edge.from, (sourceCounts.get(edge.from) ?? 0) + 1);
    const carried = [...sourceCounts].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;

    const newGroups: RuntimeGroup[] = [];
    const newNodes: RuntimeNode[] = [];
    const newEdges: WireEdge[] = [];
    for (let i = 1; i <= count; i++) {
      const suffix = `#${i}`;
      const mapGroup = (id: string) => (groupIds.has(id) ? id + suffix : id);
      newGroups.push({
        ...repeat,
        id: repeat.id + suffix,
        parentId: repeat.id,
        kind: "block",
        label: `${repeat.label} ${i}/${count}`,
        repeatCount: null,
        repeatSymbol: null,
        templateId: repeat.id,
      });
      for (const group of inner) {
        newGroups.push({
          ...group,
          id: group.id + suffix,
          parentId: group.parentId ? mapGroup(group.parentId) : null,
          templateId: group.templateId ?? group.id,
        });
      }
      for (const node of members) {
        newNodes.push({
          ...node,
          id: node.id + suffix,
          groupId: mapGroup(node.groupId),
          issues: [...node.issues],
          templateId: node.templateId ?? node.id,
        });
      }
      for (const edge of internal) {
        newEdges.push({
          ...edge,
          id: edge.id + suffix,
          from: edge.from + suffix,
          to: edge.to + suffix,
        });
      }
      for (const edge of entering) {
        const fromPrevious = i > 1 && exit !== null && edge.from === carried;
        newEdges.push({
          ...edge,
          id: edge.id + suffix,
          from: fromPrevious ? `${exit}#${i - 1}` : edge.from,
          to: edge.to + suffix,
        });
      }
    }
    for (const edge of leaving) newEdges.push({ ...edge, from: `${edge.from}#${count}` });

    const replaced = new Set([...internal, ...entering, ...leaving]);
    const innerIds = new Set(inner.map((group) => group.id));
    graph.edges = [...graph.edges.filter((edge) => !replaced.has(edge)), ...newEdges];
    spliceInPlace(graph.nodes, (node) => memberIds.has(node.id), newNodes);
    graph.groups = [...graph.groups.filter((group) => !innerIds.has(group.id)), ...newGroups];
    for (const model of graph.models) {
      model.inputNodeIds = model.inputNodeIds.map((id) => (memberIds.has(id) ? `${id}#1` : id));
      model.outputNodeIds = model.outputNodeIds.map((id) =>
        memberIds.has(id) ? `${id}#${count}` : id,
      );
    }
  }
}

/** Replaces the matching items with `replacement`, at the position of the first match. */
function spliceInPlace<T>(items: T[], matches: (item: T) => boolean, replacement: T[]): void {
  const first = items.findIndex(matches);
  const kept = items.filter((item) => !matches(item));
  const at =
    first < 0 ? kept.length : items.slice(0, first).filter((item) => !matches(item)).length;
  kept.splice(at, 0, ...replacement);
  items.length = 0;
  items.push(...kept);
}

function checkIntegrity(graph: Graph, issues: Issue[]): void {
  const nodeIds = new Set(graph.nodes.map((node) => node.id));
  const groupIds = new Set(graph.groups.map((group) => group.id));
  const modelIds = new Set(graph.models.map((model) => model.id));

  graph.edges = graph.edges.filter((edge) => {
    const ok = nodeIds.has(edge.from) && nodeIds.has(edge.to);
    if (!ok) {
      issues.push({
        code: "dangling-edge",
        message: `Edge ${edge.id} points at a node that does not exist (${edge.from} → ${edge.to}) and was dropped.`,
        severity: "error",
      });
    }
    return ok;
  });

  for (const group of graph.groups) {
    if (!modelIds.has(group.modelId)) {
      issues.push({
        code: "group-without-model",
        message: `Group ${group.id} belongs to unknown model ${group.modelId}.`,
        severity: "error",
      });
    }
    if (group.parentId !== null && !groupIds.has(group.parentId)) {
      issues.push({
        code: "group-without-parent",
        message: `Group ${group.id} has unknown parent ${group.parentId}.`,
        severity: "error",
      });
    }
  }
  for (const model of graph.models) {
    if (!groupIds.has(model.rootGroupId)) {
      issues.push({
        code: "model-without-root",
        message: `Model ${model.id} has no root group.`,
        severity: "error",
      });
    }
  }

  const inputCount = new Map<string, number>();
  for (const edge of graph.edges) inputCount.set(edge.to, (inputCount.get(edge.to) ?? 0) + 1);
  for (const node of graph.nodes) {
    if (!groupIds.has(node.groupId)) {
      node.status = "error";
      node.issues.push({
        code: "node-without-group",
        message: `Group ${node.groupId} does not exist.`,
        severity: "error",
      });
    }
    const spec = getOp(node.op);
    if (!spec) continue;
    const inputs = inputCount.get(node.id) ?? 0;
    if (inputs < spec.arity.min || (spec.arity.max !== null && inputs > spec.arity.max)) {
      const expected =
        spec.arity.max === spec.arity.min
          ? String(spec.arity.min)
          : `${spec.arity.min}${spec.arity.max === null ? " or more" : `–${spec.arity.max}`}`;
      node.status = "error";
      node.issues.push({
        code: "wrong-input-count",
        message: `${spec.title} takes ${expected} input(s) but ${inputs} are connected.`,
        severity: "error",
      });
    }
  }
}
