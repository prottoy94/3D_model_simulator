/**
 * Forge: editing a model. The document is a Draft (the same forgiving format
 * the AI writes), so every edit goes through the same expansion, shape engine
 * and verification as an extracted model. All functions here are pure: they
 * return a new document.
 */
import type { Draft, DraftNode } from "../ai/draft";
import type { RuntimeNode } from "../ir/runtime";
import { getOp, type OpFamily, type OpKind } from "../ops";

export type ForgeFramework = "keras" | "pytorch";

export type PaletteEntry = {
  op: OpKind;
  label: string;
  family: OpFamily;
  /** Settings a freshly added layer starts with. */
  defaults: Record<string, unknown>;
  hint: string;
};

/** The layers Forge offers. Every one of them can be exported as Keras and PyTorch code. */
export const PALETTE: PaletteEntry[] = [
  {
    op: "conv2d",
    label: "Conv 2D",
    family: "conv",
    defaults: { filters: 32, kernel: [3, 3], padding: "same", activation: "relu" },
    hint: "Finds local patterns with sliding filters.",
  },
  {
    op: "depthwise_conv2d",
    label: "Depthwise conv",
    family: "conv",
    defaults: { kernel: [3, 3], padding: "same" },
    hint: "One filter per channel: cheap spatial filtering.",
  },
  {
    op: "maxpool2d",
    label: "Max pool",
    family: "pool",
    defaults: { pool: [2, 2] },
    hint: "Halves the size, keeping the strongest value.",
  },
  {
    op: "avgpool2d",
    label: "Average pool",
    family: "pool",
    defaults: { pool: [2, 2] },
    hint: "Halves the size by averaging.",
  },
  {
    op: "global_avg_pool",
    label: "Global average pool",
    family: "pool",
    // Always a flat vector, in both frameworks.
    defaults: { keepDims: false },
    hint: "One number per channel: image → vector.",
  },
  {
    op: "batchnorm",
    label: "Batch norm",
    family: "norm",
    defaults: {},
    hint: "Keeps activations in a stable range.",
  },
  {
    op: "layernorm",
    label: "Layer norm",
    family: "norm",
    defaults: {},
    hint: "Normalizes each token on its own.",
  },
  {
    op: "relu",
    label: "ReLU",
    family: "activation",
    defaults: {},
    hint: "Cuts negative values to zero.",
  },
  {
    op: "gelu",
    label: "GELU",
    family: "activation",
    defaults: {},
    hint: "A smooth ReLU, common in Transformers.",
  },
  {
    op: "sigmoid",
    label: "Sigmoid",
    family: "activation",
    defaults: {},
    hint: "Squashes values into 0..1.",
  },
  {
    op: "softmax",
    label: "Softmax",
    family: "activation",
    defaults: {},
    hint: "Turns scores into probabilities.",
  },
  {
    op: "dense",
    label: "Dense",
    family: "linear",
    defaults: { units: 64, activation: "relu" },
    hint: "Every output sees every input.",
  },
  {
    op: "embedding",
    label: "Embedding",
    family: "linear",
    defaults: { vocab: 1000, dim: 64 },
    hint: "Turns integer ids into vectors.",
  },
  {
    op: "multi_head_attention",
    label: "Attention",
    family: "attention",
    defaults: { numHeads: 4, keyDim: 16 },
    hint: "Lets tokens look at each other.",
  },
  {
    op: "groupnorm",
    label: "Group norm",
    family: "norm",
    defaults: { groups: 8 },
    hint: "Normalizes channels in groups; works with any batch size.",
  },
  {
    op: "tanh",
    label: "Tanh",
    family: "activation",
    defaults: {},
    hint: "Squashes values into -1..1.",
  },
  {
    op: "lstm",
    label: "LSTM",
    family: "recurrent",
    defaults: { units: 64, returnSequences: false },
    hint: "Reads a sequence step by step with a gated memory.",
  },
  {
    op: "gru",
    label: "GRU",
    family: "recurrent",
    defaults: { units: 64, returnSequences: false },
    hint: "A lighter gated memory for sequences.",
  },
  {
    op: "upsample",
    label: "Upsample",
    family: "reshape",
    defaults: { scale: [2, 2] },
    hint: "Doubles the height and width.",
  },
  {
    op: "global_max_pool",
    label: "Global max pool",
    family: "pool",
    defaults: { keepDims: false },
    hint: "The largest value of each channel.",
  },
  {
    op: "flatten",
    label: "Flatten",
    family: "reshape",
    defaults: {},
    hint: "Unrolls a tensor into one long vector.",
  },
  {
    op: "dropout",
    label: "Dropout",
    family: "regularization",
    defaults: { rate: 0.3 },
    hint: "Randomly drops values during training.",
  },
  {
    op: "add",
    label: "Add",
    family: "merge",
    defaults: {},
    hint: "Adds two branches (a residual connection).",
  },
  {
    op: "concat",
    label: "Concatenate",
    family: "merge",
    defaults: {},
    hint: "Stacks two branches side by side.",
  },
  {
    op: "multiply",
    label: "Multiply",
    family: "merge",
    defaults: {},
    hint: "Scales one branch by another.",
  },
  {
    op: "input",
    label: "Input",
    family: "io",
    defaults: { shape: [64, 64, 3] },
    hint: "Another input to the model.",
  },
];

const pairText = (value: unknown) => (Array.isArray(value) ? value.join("×") : String(value));

/** A short label for a layer, from its settings. */
export function autoLabel(op: string, params: Record<string, unknown>): string {
  const activation =
    typeof params.activation === "string" && params.activation !== "linear"
      ? ` · ${params.activation}`
      : "";
  switch (op) {
    case "input":
      return `Input ${pairText(params.shape)}`;
    case "conv2d":
      return `Conv ${params.filters}@${pairText(params.kernel)}${activation}`;
    case "depthwise_conv2d":
      return `Depthwise ${pairText(params.kernel)}`;
    case "maxpool2d":
      return `MaxPool ${pairText(params.pool ?? [2, 2])}`;
    case "avgpool2d":
      return `AvgPool ${pairText(params.pool ?? [2, 2])}`;
    case "dense":
      return `Dense ${params.units}${activation}`;
    case "dropout":
      return `Dropout ${params.rate ?? 0.5}`;
    case "multi_head_attention":
      return `Attention · ${params.numHeads} heads`;
    case "embedding":
      return `Embedding ${params.vocab}→${params.dim}`;
    default:
      return getOp(op as OpKind)?.title ?? op;
  }
}

const clone = (doc: Draft): Draft => structuredClone(doc);
const nodesOf = (doc: Draft) => doc.models[0]!.nodes;

function freshId(doc: Draft, op: string): string {
  const taken = new Set(nodesOf(doc).map((node) => node.id));
  for (let n = 1; ; n++) {
    const id = `${op === "multi_head_attention" ? "attention" : op}_${n}`;
    if (!taken.has(id)) return id;
  }
}

/** The layer nothing reads from yet, i.e. where a new layer is appended. */
export function lastNode(doc: Draft): string | null {
  const nodes = nodesOf(doc);
  const read = new Set(nodes.flatMap((node) => node.inputs));
  const sinks = nodes.filter((node) => !read.has(node.id));
  return sinks[sinks.length - 1]?.id ?? nodes[nodes.length - 1]?.id ?? null;
}

/**
 * Adds a layer after `afterId` (or at the end). In the middle of a chain it is
 * inserted: whatever read `afterId` now reads the new layer.
 */
export function addNode(
  doc: Draft,
  op: OpKind,
  afterId: string | null,
): { doc: Draft; id: string } {
  const next = clone(doc);
  const nodes = nodesOf(next);
  const entry = PALETTE.find((item) => item.op === op);
  const id = freshId(next, op);
  const params = structuredClone(entry?.defaults ?? {});
  if (op === "input") {
    if (next.framework === "pytorch" && Array.isArray(params.shape) && params.shape.length === 3) {
      const [h, w, c] = params.shape as number[];
      params.shape = [c, h, w];
    }
    nodes.unshift({ id, op, inputs: [], params, label: autoLabel(op, params) });
    return { doc: next, id };
  }
  const after = afterId && nodes.some((node) => node.id === afterId) ? afterId : lastNode(next);
  for (const node of nodes)
    node.inputs = node.inputs.map((input) => (input === after ? id : input));
  const position = nodes.findIndex((node) => node.id === after);
  nodes.splice(position + 1, 0, {
    id,
    op,
    inputs: after ? [after] : [],
    params,
    label: autoLabel(op, params),
  });
  return { doc: next, id };
}

/** Removes a layer and reconnects what read it to what it read. */
export function removeNode(doc: Draft, id: string): Draft {
  const next = clone(doc);
  const model = next.models[0]!;
  const removed = model.nodes.find((node) => node.id === id);
  if (!removed) return doc;
  const upstream = removed.inputs[0];
  model.nodes = model.nodes.filter((node) => node.id !== id);
  for (const node of model.nodes) {
    node.inputs = [
      ...new Set(
        node.inputs.flatMap((input) => (input === id ? (upstream ? [upstream] : []) : [input])),
      ),
    ];
  }
  return next;
}

export function setParam(doc: Draft, id: string, key: string, value: unknown): Draft {
  const next = clone(doc);
  const node = nodesOf(next).find((item) => item.id === id);
  if (!node) return doc;
  if (value === undefined || value === "") delete node.params[key];
  else node.params[key] = value;
  node.label = autoLabel(node.op, node.params);
  return next;
}

/** True when `to` can be reached from `from` by following the data. */
function reaches(nodes: DraftNode[], from: string, to: string): boolean {
  const seen = new Set([from]);
  const queue = [from];
  while (queue.length > 0) {
    const current = queue.pop()!;
    if (current === to) return true;
    for (const node of nodes) {
      if (node.inputs.includes(current) && !seen.has(node.id)) {
        seen.add(node.id);
        queue.push(node.id);
      }
    }
  }
  return false;
}

/** Feeds `from`'s output into `to`. Refuses connections that would make a loop. */
export function connect(doc: Draft, from: string, to: string): Draft {
  const nodes = nodesOf(doc);
  const target = nodes.find((node) => node.id === to);
  if (
    !target ||
    from === to ||
    target.inputs.includes(from) ||
    !nodes.some((node) => node.id === from)
  )
    return doc;
  if (target.op === "input" || reaches(nodes, to, from)) return doc;
  const next = clone(doc);
  nodesOf(next)
    .find((node) => node.id === to)!
    .inputs.push(from);
  return next;
}

export function disconnect(doc: Draft, from: string, to: string): Draft {
  const next = clone(doc);
  const target = nodesOf(next).find((node) => node.id === to);
  if (!target || !target.inputs.includes(from)) return doc;
  target.inputs = target.inputs.filter((input) => input !== from);
  return next;
}

/** Switches the framework. Image inputs change between (H, W, C) and (C, H, W). */
export function setFramework(doc: Draft, framework: ForgeFramework): Draft {
  if (doc.framework === framework) return doc;
  const next = clone(doc);
  next.framework = framework;
  for (const node of nodesOf(next)) {
    if (node.op === "input" && Array.isArray(node.params.shape) && node.params.shape.length === 3) {
      const [a, b, c] = node.params.shape as number[];
      node.params.shape = framework === "pytorch" ? [c, a, b] : [b, c, a];
      node.label = autoLabel(node.op, node.params);
    }
  }
  return next;
}

export function renameModel(doc: Draft, name: string): Draft {
  const next = clone(doc);
  next.models[0]!.name = name;
  next.title = name;
  return next;
}

// ── Templates ─────────────────────────────────────────────────────────

type Step = [id: string, op: OpKind, params?: Record<string, unknown>, inputs?: string[]];

function chain(name: string, summary: string, steps: Step[]): Draft {
  let previous: string | null = null;
  const nodes = steps.map(([id, op, given, inputs]): DraftNode => {
    const params =
      given ?? structuredClone(PALETTE.find((entry) => entry.op === op)?.defaults ?? {});
    const node: DraftNode = {
      id,
      op,
      params,
      inputs: inputs ?? (previous ? [previous] : []),
      label: autoLabel(op, params),
    };
    previous = id;
    return node;
  });
  return {
    title: name,
    summary,
    framework: "keras",
    symbols: [],
    warnings: [],
    models: [{ id: "forge", name, groups: [], nodes, outputs: [] }],
  };
}

const relu = (filters: number): Record<string, unknown> => ({
  filters,
  kernel: [3, 3],
  padding: "same",
  activation: "relu",
});

export const TEMPLATES: Array<{
  id: string;
  label: string;
  description: string;
  build: () => Draft;
}> = [
  {
    id: "empty",
    label: "Empty",
    description: "Just an input. Add layers from the palette.",
    build: () =>
      chain("MyModel", "A model built in Forge.", [["input_1", "input", { shape: [64, 64, 3] }]]),
  },
  {
    id: "small-cnn",
    label: "Small CNN",
    description: "Two conv + pool stages and a pooled classifier.",
    build: () =>
      chain("SmallCNN", "Two convolution stages and a pooled classifier head.", [
        ["input_1", "input", { shape: [64, 64, 3] }],
        ["conv2d_1", "conv2d", relu(32)],
        ["maxpool2d_1", "maxpool2d", { pool: [2, 2] }],
        ["conv2d_2", "conv2d", relu(64)],
        ["maxpool2d_2", "maxpool2d", { pool: [2, 2] }],
        ["global_avg_pool_1", "global_avg_pool"],
        ["dropout_1", "dropout", { rate: 0.3 }],
        ["dense_1", "dense", { units: 10, activation: "softmax" }],
      ]),
  },
  {
    id: "residual",
    label: "Residual block",
    description: "A stem, one residual block with a skip connection, and a head.",
    build: () =>
      chain("ResidualNet", "A convolution stem, a residual block and a pooled classifier.", [
        ["input_1", "input", { shape: [64, 64, 3] }],
        ["conv2d_1", "conv2d", relu(32)],
        ["conv2d_2", "conv2d", relu(32)],
        ["batchnorm_1", "batchnorm"],
        ["conv2d_3", "conv2d", { filters: 32, kernel: [3, 3], padding: "same" }],
        ["add_1", "add", {}, ["conv2d_3", "conv2d_1"]],
        ["relu_1", "relu"],
        ["global_avg_pool_1", "global_avg_pool"],
        ["dense_1", "dense", { units: 10, activation: "softmax" }],
      ]),
  },
  {
    id: "mlp",
    label: "MLP",
    description: "Three dense layers on a flat vector.",
    build: () =>
      chain("MLP", "A multilayer perceptron on a 784-number input.", [
        ["input_1", "input", { shape: [784] }],
        ["dense_1", "dense", { units: 256, activation: "relu" }],
        ["dropout_1", "dropout", { rate: 0.3 }],
        ["dense_2", "dense", { units: 64, activation: "relu" }],
        ["dense_3", "dense", { units: 10, activation: "softmax" }],
      ]),
  },
  {
    id: "transformer",
    label: "Transformer encoder",
    description: "One pre-norm encoder block over 32 tokens, then a classifier.",
    build: () =>
      chain("TinyTransformer", "One Transformer encoder block over a sequence of 32 tokens.", [
        ["input_1", "input", { shape: [32, 64] }],
        ["layernorm_1", "layernorm"],
        [
          "attention_1",
          "multi_head_attention",
          { numHeads: 4, keyDim: 16 },
          ["layernorm_1", "layernorm_1"],
        ],
        ["add_1", "add", {}, ["attention_1", "input_1"]],
        ["layernorm_2", "layernorm"],
        ["dense_1", "dense", { units: 128, activation: "gelu" }],
        ["dense_2", "dense", { units: 64 }],
        ["add_2", "add", {}, ["dense_2", "add_1"]],
        ["global_avg_pool_1", "global_avg_pool"],
        ["dense_3", "dense", { units: 2, activation: "softmax" }],
      ]),
  },
];

// ── Fix suggestions ───────────────────────────────────────────────────

/** A plain-language way out of a shape or wiring problem, when there is an obvious one. */
export function suggestFix(node: RuntimeNode): string | null {
  const text = node.issues.map((issue) => `${issue.code} ${issue.message}`).join(" ");
  if (/wrong-input-count/.test(text)) {
    return ["add", "concat", "multiply"].includes(node.op)
      ? "This layer combines two branches. Connect a second layer into it: drag from that layer's bottom dot to this card in the 2D view."
      : "Connect exactly one layer into this one, or remove the extra connection.";
  }
  if (/does not fit/.test(text)) {
    return 'The window is larger than what is left of the image. Set padding to "same", use a smaller kernel, or remove an earlier pooling layer.';
  }
  if (/only allows padding="same" with stride 1/.test(text)) {
    return 'PyTorch cannot pad "same" when the stride is above 1. Set the stride to 1, or use explicit padding such as 1.';
  }
  if (/must be an image tensor/.test(text)) {
    return "This layer needs an image (height, width, channels), but it comes after a layer that already flattened or pooled it. Move it before that layer.";
  }
  if (/shapes do not line up/.test(text)) {
    return "Both branches must have the same shape to be combined. Give them the same number of filters and the same stride, or put a 1×1 convolution on the shorter branch.";
  }
  if (/inputs must match on every axis/.test(text)) {
    return "Concatenated branches must have the same height and width. Use the same stride and padding on both.";
  }
  if (/needs a sequence|expected tokens/.test(text)) {
    return "This layer works on a sequence of tokens (tokens, size). Feed it a token input, not an image or a flat vector.";
  }
  if (/bad-param/.test(text))
    return "One of this layer's settings is missing or not valid. Check the fields above.";
  if (/input-unknown/.test(text))
    return "Fix the layer before this one first; this one will then compute.";
  return null;
}
