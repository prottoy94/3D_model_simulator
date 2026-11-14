/** Input, normalization, activation and regularization ops. */
import { z } from "zod";
import {
  BATCH,
  elementsPerSample,
  featureDim,
  lastDim,
  layoutForRank,
  ShapeError,
  shapeText,
  type Dim,
} from "../shapes/tensor";
import type { OpKind } from "./kinds";
import { defineOp, NO_PARAMS, type AnyOpSpec, type VisualizerId } from "./types";

const inputParams = z.object({
  /** Shape without the batch. A negative entry is a size only known at run time. */
  shape: z.array(z.number().int()).min(1),
  layout: z.enum(["NHWC", "NCHW", "NLC", "NCL", "NC", "NL", "any"]).optional(),
  /** Name shown for run-time sizes, e.g. "R" for a variable number of RoIs. */
  dynamicDim: z.string().default("N"),
});

type InputParams = z.infer<typeof inputParams>;

export const input = defineOp<InputParams>({
  op: "input",
  family: "io",
  title: "Input",
  aliases: { keras: ["Input", "InputLayer"] },
  params: inputParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: { keras: { positional: ["shape"], keys: { shape: "shape" } } },
  arity: { min: 0, max: 0 },
  inferShape: (_inputs, p, ctx) => {
    const dims: Dim[] = [
      BATCH,
      ...p.shape.map((size): Dim => (size < 0 ? { sym: p.dynamicDim } : size)),
    ];
    return { dims, layout: p.layout ?? layoutForRank(dims.length, ctx.framework) };
  },
  countParams: () => NO_PARAMS,
  flops: () => 0,
  glyph: "slab",
  visualizer: "none",
  explain: (_p, io) =>
    `The data enters the model here with shape ${shapeText(io.out)}. B is the batch size.`,
  codegen: {
    keras: (p) =>
      `keras.Input(shape=(${p.shape.map((size) => (size < 0 ? "None" : size)).join(", ")}))`,
    pytorch: (p) => `# input tensor of shape (B, ${p.shape.join(", ")})`,
  },
});

// ── Normalization ─────────────────────────────────────────────────────

const normParams = z.object({
  epsilon: z.number().positive().default(1e-3),
  momentum: z.number().default(0.99),
  /** β (shift) and γ (scale) are learned when true. */
  center: z.boolean().default(true),
  scale: z.boolean().default(true),
});

type NormParams = z.infer<typeof normParams>;

export const batchnorm = defineOp<NormParams>({
  op: "batchnorm",
  family: "norm",
  title: "Batch Normalization",
  aliases: { keras: ["BatchNormalization"], pytorch: ["nn.BatchNorm2d", "nn.BatchNorm1d"] },
  params: normParams,
  defaults: { keras: { epsilon: 1e-3, momentum: 0.99 }, pytorch: { epsilon: 1e-5, momentum: 0.1 } },
  signatures: {
    keras: {
      positional: ["axis", "momentum", "epsilon", "center", "scale"],
      keys: { momentum: "momentum", epsilon: "epsilon", center: "center", scale: "scale" },
    },
    pytorch: {
      positional: ["num_features", "eps", "momentum", "affine"],
      keys: { eps: "epsilon", momentum: "momentum" },
    },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([shape]) => shape!,
  // Keras counts the moving mean and variance as non-trainable params.
  // PyTorch keeps them as buffers, which are not params at all.
  countParams: ([shape], p, ctx) => {
    const c = featureDim(shape!, "channels");
    const learned = (p.center ? c : 0) + (p.scale ? c : 0);
    return { trainable: learned, nonTrainable: ctx.framework === "keras" ? 2 * c : 0 };
  },
  flops: ([shape]) => 2 * elementsPerSample(shape!),
  glyph: "plate",
  visualizer: "norm-histogram",
  explain: (_p, io) => {
    const c = featureDim(io.in[0]!);
    return `Each of the ${c} channels is shifted and scaled to zero mean and unit variance across the batch, then given a learned scale (γ) and shift (β). This keeps activations in a stable range.`;
  },
  codegen: {
    keras: () => "layers.BatchNormalization()",
    pytorch: (_p, io) =>
      `nn.BatchNorm${io.in[0]!.dims.length === 4 ? "2d" : "1d"}(${featureDim(io.in[0]!)})`,
  },
});

export const layernorm = defineOp<NormParams>({
  op: "layernorm",
  family: "norm",
  title: "Layer Normalization",
  aliases: { keras: ["LayerNormalization"], pytorch: ["nn.LayerNorm"] },
  params: normParams,
  defaults: { keras: { epsilon: 1e-3 }, pytorch: { epsilon: 1e-5 } },
  signatures: {
    keras: {
      positional: ["axis", "epsilon", "center", "scale"],
      keys: { epsilon: "epsilon", center: "center", scale: "scale" },
    },
    pytorch: { positional: ["normalized_shape", "eps"], keys: { eps: "epsilon" } },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([shape]) => shape!,
  countParams: ([shape], p) => {
    const d = lastDim(shape!);
    return { trainable: (p.center ? d : 0) + (p.scale ? d : 0), nonTrainable: 0 };
  },
  flops: ([shape]) => 2 * elementsPerSample(shape!),
  glyph: "plate",
  visualizer: "norm-histogram",
  explain: (_p, io) =>
    `Each token's ${lastDim(io.in[0]!)} values are normalized on their own (no batch statistics), then scaled and shifted by learned γ and β.`,
  codegen: {
    keras: (p) => `layers.LayerNormalization(epsilon=${p.epsilon})`,
    pytorch: (p, io) => `nn.LayerNorm(${lastDim(io.in[0]!)}, eps=${p.epsilon})`,
  },
});

// ── Activations ───────────────────────────────────────────────────────

const activationParams = z.object({ axis: z.number().int().default(-1) });

type ActivationParams = z.infer<typeof activationParams>;

type ActivationInfo = {
  op: OpKind;
  title: string;
  keras: string[];
  pytorch: string;
  functional: string[];
  visualizer: VisualizerId;
  explain: string;
  kerasCode: string;
};

function activationOp(info: ActivationInfo): AnyOpSpec {
  return defineOp<ActivationParams>({
    op: info.op,
    family: "activation",
    title: info.title,
    aliases: { keras: info.keras, pytorch: [info.pytorch], functional: info.functional },
    params: activationParams,
    defaults: { keras: {}, pytorch: {} },
    signatures: {
      keras: { positional: [], keys: { axis: "axis" } },
      pytorch: { positional: [], keys: { dim: "axis" } },
    },
    arity: { min: 1, max: 1 },
    inferShape: ([shape]) => {
      if (!shape) throw new ShapeError("missing input");
      return shape;
    },
    countParams: () => NO_PARAMS,
    flops: ([shape]) => elementsPerSample(shape!),
    glyph: "plate",
    visualizer: info.visualizer,
    explain: () => info.explain,
    codegen: { keras: () => info.kerasCode, pytorch: () => `${info.pytorch}()` },
  });
}

export const activationOps = [
  activationOp({
    op: "relu",
    title: "ReLU",
    keras: ["ReLU"],
    pytorch: "nn.ReLU",
    functional: ["F.relu", "tf.nn.relu", "torch.relu"],
    visualizer: "activation-curve",
    explain: "Negative values become 0 and positive values pass through unchanged. No weights.",
    kerasCode: "layers.ReLU()",
  }),
  activationOp({
    op: "gelu",
    title: "GELU",
    keras: ["GELU"],
    pytorch: "nn.GELU",
    functional: ["F.gelu", "tf.nn.gelu"],
    visualizer: "activation-curve",
    explain:
      "A smooth version of ReLU: small negative values are damped instead of cut to 0. No weights.",
    kerasCode: 'layers.Activation("gelu")',
  }),
  activationOp({
    op: "sigmoid",
    title: "Sigmoid",
    keras: ["Sigmoid"],
    pytorch: "nn.Sigmoid",
    functional: ["torch.sigmoid", "F.sigmoid", "tf.sigmoid"],
    visualizer: "activation-curve",
    explain:
      "Squashes every value into the range 0 to 1, so it can be read as a probability or a gate.",
    kerasCode: 'layers.Activation("sigmoid")',
  }),
  activationOp({
    op: "softmax",
    title: "Softmax",
    keras: ["Softmax"],
    pytorch: "nn.Softmax",
    functional: ["F.softmax", "tf.nn.softmax", "torch.softmax"],
    visualizer: "softmax-bars",
    explain: "Turns the scores into probabilities that are all positive and add up to 1.",
    kerasCode: "layers.Softmax()",
  }),
];

// ── Regularization ────────────────────────────────────────────────────

const dropoutParams = z.object({ rate: z.number().min(0).max(1).default(0.5) });

type DropoutParams = z.infer<typeof dropoutParams>;

export const dropout = defineOp<DropoutParams>({
  op: "dropout",
  family: "regularization",
  title: "Dropout",
  aliases: { keras: ["Dropout"], pytorch: ["nn.Dropout"], functional: ["F.dropout"] },
  params: dropoutParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {
    keras: { positional: ["rate"], keys: { rate: "rate" } },
    pytorch: { positional: ["p"], keys: { p: "rate" } },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([shape]) => shape!,
  countParams: () => NO_PARAMS,
  flops: () => 0,
  glyph: "ghost",
  visualizer: "dropout-mask",
  explain: (p) =>
    `During training, ${Math.round(p.rate * 100)}% of the values are set to 0 at random so the model cannot rely on any single one. At inference it does nothing.`,
  codegen: { keras: (p) => `layers.Dropout(${p.rate})`, pytorch: (p) => `nn.Dropout(${p.rate})` },
});

export const basicOps = [input, batchnorm, layernorm, ...activationOps, dropout];
