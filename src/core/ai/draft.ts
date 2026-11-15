/**
 * The Draft: the compact JSON an LLM writes (ADR-015). It is deliberately
 * forgiving: few required fields, params as a plain object, connections as an
 * `inputs` list on each node. `draftToWire` expands it into strict Wire IR,
 * which the engine then normalizes and verifies like any other IR.
 *
 * Small free models make far fewer mistakes with this than with Wire IR itself.
 */
import { z } from "zod";
import { ref, WireBuilder } from "../ir/builder";
import type { CodeRef, ParamValue, WireGroup, WireIR } from "../ir/wire";
import { getOp, OP_KINDS, opForLayerName, type OpKind } from "../ops";
import { resolveZoo } from "../zoo";

/** [cell, firstLine, lastLine], lines 1-based within the cell. */
const linesSchema = z.tuple([z.number().int(), z.number().int(), z.number().int()]).nullish();

const draftNodeSchema = z.object({
  id: z.string().min(1),
  op: z.string().min(1),
  label: z.string().nullish(),
  group: z.string().nullish(),
  inputs: z.array(z.string()).default([]),
  params: z.record(z.string(), z.unknown()).default({}),
  lines: linesSchema,
  /** The class or function name in the user's code, when the op wraps one. */
  className: z.string().nullish(),
  frozen: z.boolean().nullish(),
  timeDistributed: z.boolean().nullish(),
  outputShape: z.array(z.union([z.number(), z.string(), z.null()])).nullish(),
  explanation: z.string().nullish(),
});

const draftGroupSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  kind: z.string().nullish(),
  parent: z.string().nullish(),
  /** How many times the block repeats: a number or the name of a symbol. */
  repeat: z.union([z.number().int(), z.string()]).nullish(),
  lines: linesSchema,
  description: z.string().nullish(),
});

const draftModelSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  description: z.string().nullish(),
  primary: z.boolean().nullish(),
  lines: linesSchema,
  groups: z.array(draftGroupSchema).default([]),
  nodes: z.array(draftNodeSchema).min(1),
  outputs: z.array(z.string()).default([]),
});

const draftSymbolSchema = z.object({
  name: z.string().min(1),
  value: z.unknown(),
  source: z.string().nullish(),
  note: z.string().nullish(),
});

export const draftSchema = z.object({
  title: z.string().default("Model"),
  summary: z.string().default(""),
  framework: z.string().default("keras"),
  task: z.string().nullish(),
  symbols: z.array(draftSymbolSchema).default([]),
  models: z.array(draftModelSchema).min(1),
  warnings: z.array(z.string()).default([]),
});

export type Draft = z.infer<typeof draftSchema>;
export type DraftNode = z.infer<typeof draftNodeSchema>;

export const draftPatchSchema = z.object({
  /**
   * Nodes to change or add. For an existing id only the fields given are
   * changed; a new id needs at least `op`. `model` picks the model for new ones.
   */
  nodes: z
    .array(
      // Written out without defaults: a field the patch leaves out must stay undefined.
      z.object({
        id: z.string().min(1),
        model: z.string().nullish(),
        op: z.string().min(1).optional(),
        label: z.string().nullish(),
        group: z.string().nullish(),
        inputs: z.array(z.string()).optional(),
        params: z.record(z.string(), z.unknown()).optional(),
        lines: linesSchema,
        className: z.string().nullish(),
        frozen: z.boolean().nullish(),
        timeDistributed: z.boolean().nullish(),
        outputShape: z.array(z.union([z.number(), z.string(), z.null()])).nullish(),
        explanation: z.string().nullish(),
      }),
    )
    .default([]),
  remove: z.array(z.string()).default([]),
  symbols: z.array(draftSymbolSchema).default([]),
});

export type DraftPatch = z.infer<typeof draftPatchSchema>;

export type ParseResult<T> = { ok: true; value: T } | { ok: false; errors: string[] };

/** Pulls the JSON object out of a model reply that may be wrapped in prose or a code fence. */
export function extractJson(text: string): unknown {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end <= start) throw new Error("the reply contains no JSON object");
  return JSON.parse(text.slice(start, end + 1));
}

function parseWith<T>(schema: z.ZodType<T>, text: string): ParseResult<T> {
  let json: unknown;
  try {
    json = extractJson(text);
  } catch (error) {
    return { ok: false, errors: [error instanceof Error ? error.message : "invalid JSON"] };
  }
  const result = schema.safeParse(json);
  if (result.success) return { ok: true, value: result.data };
  return {
    ok: false,
    errors: result.error.issues
      .slice(0, 12)
      .map((issue) => `${issue.path.join(".") || "(root)"}: ${issue.message}`),
  };
}

export const parseDraft = (text: string) => parseWith(draftSchema, text);
export const parseDraftPatch = (text: string) => parseWith(draftPatchSchema, text);

/** Applies a repair patch. Returns a new draft; the input is not changed. */
export function applyDraftPatch(draft: Draft, patch: DraftPatch): Draft {
  const next = structuredClone(draft);
  const removed = new Set(patch.remove);
  for (const model of next.models) {
    model.nodes = model.nodes.filter((node) => !removed.has(node.id));
    for (const node of model.nodes) node.inputs = node.inputs.filter((id) => !removed.has(id));
    model.outputs = model.outputs.filter((id) => !removed.has(id));
  }
  for (const { model: modelId, ...change } of patch.nodes) {
    const home =
      next.models.find((model) => model.nodes.some((existing) => existing.id === change.id)) ??
      next.models.find((model) => model.id === modelId) ??
      next.models[0]!;
    const existing = home.nodes.find((node) => node.id === change.id);
    // Fields the patch leaves out (often `lines` and `group`) keep their old values.
    const given = Object.fromEntries(
      Object.entries(change).filter(([, value]) => value !== undefined),
    );
    if (existing) Object.assign(existing, given);
    else if (change.op) home.nodes.push(draftNodeSchema.parse(given));
  }
  for (const symbol of patch.symbols) {
    const existing = next.symbols.find((entry) => entry.name === symbol.name);
    if (existing) Object.assign(existing, symbol);
    else next.symbols.push(symbol);
  }
  return next;
}

// ── Draft → Wire IR ───────────────────────────────────────────────────

const FRAMEWORKS: ReadonlyArray<WireIR["framework"]> = [
  "keras",
  "pytorch",
  "tensorflow",
  "jax",
  "sklearn",
  "huggingface",
  "ultralytics",
  "mixed",
  "other",
];
const TASKS: ReadonlyArray<WireIR["task"]> = [
  "classification",
  "detection",
  "segmentation",
  "regression",
  "generation",
  "sequence-labeling",
  "clustering",
  "embedding",
  "multi-task",
  "other",
];
const GROUP_KINDS: ReadonlyArray<WireGroup["kind"]> = [
  "stage",
  "block",
  "repeat",
  "module",
  "backbone",
  "head",
  "branch",
];
const PROVENANCE = ["literal", "computed", "from-output", "assumed"] as const;

const oneOf = <T extends string>(options: ReadonlyArray<T>, value: unknown, fallback: T): T =>
  options.find((option) => option === String(value ?? "").toLowerCase()) ?? fallback;

/**
 * Evaluates small arithmetic over symbols, e.g. "num_classes + 1" or
 * "IMG // 32". Returns undefined for anything else. Nothing is executed:
 * this is a tiny parser for numbers, names, + - * / // ** and parentheses.
 */
export function evaluateExpression(
  text: string,
  symbols: ReadonlyMap<string, unknown>,
): number | undefined {
  const tokens = text.match(/\*\*|\/\/|[-+*/()]|\d+(?:\.\d+)?|[A-Za-z_]\w*|\S/g) ?? [];
  let position = 0;
  const peek = () => tokens[position];
  const fail = () => {
    throw new Error("not arithmetic");
  };

  const atom = (): number => {
    const token = tokens[position++];
    if (token === undefined) return fail();
    if (token === "(") {
      const value = sum();
      if (tokens[position++] !== ")") fail();
      return value;
    }
    if (token === "-") return -atom();
    if (/^\d/.test(token)) return Number(token);
    const value = symbols.get(token);
    return typeof value === "number" ? value : fail();
  };
  const power = (): number => {
    const base = atom();
    if (peek() !== "**") return base;
    position++;
    return base ** power();
  };
  const product = (): number => {
    let value = power();
    for (let op = peek(); op === "*" || op === "/" || op === "//"; op = peek()) {
      position++;
      const right = power();
      value = op === "*" ? value * right : op === "/" ? value / right : Math.floor(value / right);
    }
    return value;
  };
  const sum = (): number => {
    let value = product();
    for (let op = peek(); op === "+" || op === "-"; op = peek()) {
      position++;
      const right = product();
      value = op === "+" ? value + right : value - right;
    }
    return value;
  };

  try {
    const value = sum();
    return position === tokens.length && Number.isFinite(value) ? value : undefined;
  } catch {
    return undefined;
  }
}

function toParamValue(
  value: unknown,
  symbols: ReadonlyMap<string, unknown>,
): ParamValue | undefined {
  if (value === null || value === undefined) return undefined;
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "string") {
    // A bare symbol name is kept, so the symbol stays editable. Arithmetic is worked out here.
    if (symbols.has(value)) return value;
    return /[-+*/]/.test(value) ? (evaluateExpression(value, symbols) ?? value) : value;
  }
  if (Array.isArray(value)) {
    // `None` in a shape means "known at run time"; names and arithmetic inside a list are resolved.
    const numbers = value.map((item) => {
      if (item === null) return -1;
      if (typeof item === "string") return evaluateExpression(item, symbols) ?? item;
      return item as unknown;
    });
    if (numbers.every((item): item is number => typeof item === "number")) return numbers;
  }
  return JSON.stringify(value);
}

const toRefs = (lines: DraftNode["lines"], role: CodeRef["role"]): CodeRef[] =>
  lines ? [ref(lines[0], lines[1], Math.max(lines[1], lines[2]), role)] : [];

/** Maps whatever the model called the layer onto a catalog op, or "custom". */
function resolveOp(name: string, framework: "keras" | "pytorch"): OpKind | null {
  const lower = name
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  const exact = OP_KINDS.find((kind) => kind === lower);
  if (exact) return exact;
  return opForLayerName(name, framework) ?? opForLayerName(name) ?? null;
}

export function draftToWire(draft: Draft): WireIR {
  const framework = oneOf(FRAMEWORKS, draft.framework, "keras");
  const engine =
    framework === "pytorch" || framework === "ultralytics" || framework === "huggingface"
      ? "pytorch"
      : "keras";
  const b = new WireBuilder({
    title: draft.title,
    summary: draft.summary,
    framework,
    task: oneOf(TASKS, draft.task, "classification"),
  });
  for (const symbol of draft.symbols) {
    b.symbol(symbol.name, symbol.value, {
      provenance: oneOf(PROVENANCE, symbol.source, "literal"),
      note: symbol.note ?? "",
    });
  }
  for (const warning of draft.warnings) b.warn(warning);
  const symbols = new Map(draft.symbols.map((symbol) => [symbol.name, symbol.value]));

  // Ids only have to be unique inside a model in the draft; the IR needs them unique overall.
  const many = draft.models.length > 1;
  draft.models.forEach((model, modelIndex) => {
    const scope = (id: string) => (many ? `${id}@${modelIndex + 1}` : id);
    const modelId = `m_${modelIndex + 1}`;
    b.model(modelId, model.name, model.description ?? "", toRefs(model.lines, "definition"));
    const root = `g_${modelIndex + 1}`;

    const groupIds = new Set(model.groups.map((group) => group.id));
    for (const group of model.groups) {
      const isRepeat = group.repeat !== null && group.repeat !== undefined;
      b.group(scope(group.id), group.label, {
        parent:
          group.parent && groupIds.has(group.parent) && group.parent !== group.id
            ? scope(group.parent)
            : root,
        kind: isRepeat ? "repeat" : oneOf(GROUP_KINDS, group.kind, "block"),
        repeatCount: typeof group.repeat === "number" ? group.repeat : null,
        repeatSymbol: typeof group.repeat === "string" ? group.repeat : null,
        refs: toRefs(group.lines, isRepeat ? "loop" : "definition"),
        description: group.description ?? "",
      });
    }

    const nodeIds = new Set(model.nodes.map((node) => node.id));
    for (const node of model.nodes) {
      const params: Record<string, ParamValue> = {};
      for (const [key, value] of Object.entries(node.params)) {
        const converted = toParamValue(value, symbols);
        if (converted !== undefined) params[key] = converted;
      }
      let op = resolveOp(node.op, engine);
      let customOpName = node.className ?? null;
      if (!op) {
        customOpName ??= node.op;
        op = "custom";
      }
      const declared = getOp(op)?.declared === true;
      const claimed = node.outputShape?.map((dim) => (dim === null ? "N" : dim)) ?? null;

      if (op === "backbone") {
        // Fill in what the zoo knows; anything the model already stated is kept.
        const name = String(params.zoo ?? params.name ?? node.className ?? node.label ?? "");
        const size =
          Array.isArray(params.inputSize) && params.inputSize.length >= 2
            ? (params.inputSize as number[])
            : [224, 224];
        const facts = name ? resolveZoo(name, [size[0]!, size[1]!], engine) : null;
        delete params.zoo;
        delete params.name;
        delete params.inputSize;
        if (facts) {
          params.outputShape = facts.outputShape;
          params.paramCount = facts.paramCount;
          params.nonTrainableCount = facts.nonTrainableCount;
          params.description = facts.entry.description;
          customOpName = facts.entry.key;
        }
      }
      // A declared layer with no stated shape falls back to what the model believes it outputs.
      if (declared && params.outputShape === undefined && claimed && claimed.length > 1) {
        params.outputShape = claimed.slice(1).map((dim) => (typeof dim === "number" ? dim : -1));
      }
      if (node.timeDistributed) params.timeDistributed = true;

      const spec = getOp(op);
      const hasWeights =
        spec !== undefined &&
        !["io", "pool", "activation", "reshape", "merge", "regularization"].includes(spec.family);
      b.node(scope(node.id), op, params, {
        group: node.group && groupIds.has(node.group) ? scope(node.group) : root,
        from: node.inputs.filter((id) => nodeIds.has(id) && id !== node.id).map(scope),
        label: node.label ?? node.className ?? spec?.title ?? node.op,
        refs: toRefs(node.lines, "declare"),
        explanation: node.explanation ?? "",
        customOpName: customOpName ?? undefined,
        trainable: node.frozen ? "no" : hasWeights ? "yes" : "n/a",
        claimed: declared ? undefined : (claimed ?? undefined),
        confidence: "medium",
      });
    }

    const used = new Set(model.nodes.flatMap((node) => node.inputs));
    const outputs = model.outputs.filter((id) => nodeIds.has(id));
    // If the model did not name its outputs, they are the nodes nothing reads from.
    b.outputs(
      ...(outputs.length > 0
        ? outputs
        : model.nodes.filter((node) => !used.has(node.id)).map((node) => node.id)
      ).map(scope),
    );
  });
  return b.build();
}
