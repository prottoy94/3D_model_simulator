/**
 * Phase 8 layers: group norm, more activations, resizing and reductions,
 * attention modules (squeeze-excite, CBAM channel and spatial attention),
 * recurrent layers and anchor generation.
 */
import { z } from "zod";
import {
  BATCH,
  elementsPerSample,
  featureDim,
  image,
  imageLike,
  isImage,
  known,
  lastDim,
  layoutForRank,
  resolveAxis,
  ShapeError,
  shapeText,
  type Dim,
  type TensorShape,
} from "../shapes/tensor";
import { activationOp } from "./basic";
import { defineOp, NO_PARAMS, pair, posInt, tuple, type AnyOpSpec } from "./types";

// ── Normalization and activations ─────────────────────────────────────

const groupnormParams = z.object({
  groups: posInt.default(32),
  epsilon: z.number().positive().default(1e-5),
});
type GroupnormParams = z.infer<typeof groupnormParams>;

export const groupnorm = defineOp<GroupnormParams>({
  op: "groupnorm",
  family: "norm",
  title: "Group Normalization",
  aliases: { keras: ["GroupNormalization"], pytorch: ["nn.GroupNorm"] },
  params: groupnormParams,
  defaults: { keras: { epsilon: 1e-3 }, pytorch: { epsilon: 1e-5 } },
  signatures: {
    keras: { positional: ["groups"], keys: { groups: "groups", epsilon: "epsilon" } },
    pytorch: {
      positional: ["num_groups", "num_channels", "eps"],
      keys: { num_groups: "groups", eps: "epsilon" },
    },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([shape], p) => {
    const c = featureDim(shape!, "channels");
    if (c % p.groups !== 0)
      throw new ShapeError(`${c} channels cannot be split into ${p.groups} equal groups`);
    return shape!;
  },
  countParams: ([shape]) => ({ trainable: 2 * featureDim(shape!), nonTrainable: 0 }),
  flops: ([shape]) => 2 * elementsPerSample(shape!),
  glyph: "plate",
  visualizer: "norm-histogram",
  explain: (p, io) =>
    `The ${featureDim(io.in[0]!)} channels are split into ${p.groups} groups, and each group is normalized on its own for every sample. Unlike batch norm it does not depend on the batch size.`,
  codegen: {
    keras: (p) => `layers.GroupNormalization(groups=${p.groups})`,
    pytorch: (p, io) => `nn.GroupNorm(${p.groups}, ${featureDim(io.in[0]!)})`,
  },
});

const moreActivations = [
  activationOp({
    op: "tanh",
    title: "Tanh",
    keras: ["Tanh"],
    pytorch: "nn.Tanh",
    functional: ["torch.tanh", "F.tanh", "tf.tanh"],
    visualizer: "activation-curve",
    explain: "Squashes every value into the range -1 to 1, keeping its sign.",
    kerasCode: 'layers.Activation("tanh")',
  }),
  activationOp({
    op: "silu",
    title: "SiLU (Swish)",
    keras: ["Swish"],
    pytorch: "nn.SiLU",
    functional: ["F.silu", "tf.nn.silu", "tf.nn.swish"],
    visualizer: "activation-curve",
    explain:
      "Each value is multiplied by its own sigmoid: smooth, and slightly negative for small negative inputs.",
    kerasCode: 'layers.Activation("silu")',
  }),
  activationOp({
    op: "leaky_relu",
    title: "Leaky ReLU",
    keras: ["LeakyReLU"],
    pytorch: "nn.LeakyReLU",
    functional: ["F.leaky_relu", "tf.nn.leaky_relu"],
    visualizer: "activation-curve",
    explain:
      "Like ReLU, but negative values are scaled down instead of cut to zero, so they still carry a gradient.",
    kerasCode: "layers.LeakyReLU()",
  }),
  activationOp({
    op: "hardswish",
    title: "Hard Swish",
    keras: ["HardSwish"],
    pytorch: "nn.Hardswish",
    functional: ["F.hardswish"],
    visualizer: "activation-curve",
    explain: "A cheap, piecewise-linear version of SiLU used in mobile networks.",
    kerasCode: 'layers.Activation("hard_swish")',
  }),
];

// ── Pooling, resizing, reductions ─────────────────────────────────────

const keepDimsParams = z.object({ keepDims: z.boolean().default(false) });
type KeepDimsParams = z.infer<typeof keepDimsParams>;

export const globalMaxPool = defineOp<KeepDimsParams>({
  op: "global_max_pool",
  family: "pool",
  title: "Global Max Pooling",
  aliases: {
    keras: ["GlobalMaxPooling2D", "GlobalMaxPooling1D", "GlobalMaxPool2D", "GlobalMaxPool1D"],
    pytorch: ["nn.AdaptiveMaxPool2d"],
    functional: ["F.adaptive_max_pool2d"],
  },
  params: keepDimsParams,
  defaults: { keras: {}, pytorch: { keepDims: true } },
  signatures: {
    keras: { positional: [], keys: { keepdims: "keepDims" } },
    pytorch: { positional: ["output_size"], keys: {} },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([input], p): TensorShape => {
    const shape = input!;
    const batch = shape.dims[0] ?? BATCH;
    const c = featureDim(shape);
    if (isImage(shape))
      return p.keepDims ? imageLike(shape, 1, 1, c) : { dims: [batch, c], layout: "NC" };
    if (shape.dims.length === 3) {
      if (!p.keepDims) return { dims: [batch, c], layout: "NC" };
      return shape.layout === "NCL"
        ? { dims: [batch, c, 1], layout: "NCL" }
        : { dims: [batch, 1, c], layout: "NLC" };
    }
    throw new ShapeError("global pooling needs an image or a sequence");
  },
  countParams: () => NO_PARAMS,
  flops: ([input]) => elementsPerSample(input!),
  glyph: "funnel",
  visualizer: "global-pool",
  explain: (_p, io) =>
    `Each of the ${featureDim(io.in[0]!)} feature maps is reduced to its single largest value. No weights.`,
  codegen: {
    keras: (_p, io) =>
      isImage(io.in[0]!) ? "layers.GlobalMaxPooling2D()" : "layers.GlobalMaxPooling1D()",
    pytorch: (p) =>
      p.keepDims
        ? "nn.AdaptiveMaxPool2d(1)"
        : "nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())",
  },
});

const adaptiveParams = z.object({ output: pair });
type AdaptiveParams = z.infer<typeof adaptiveParams>;

export const adaptiveAvgPool2d = defineOp<AdaptiveParams>({
  op: "adaptive_avg_pool2d",
  family: "pool",
  title: "Adaptive Average Pooling",
  aliases: {},
  params: adaptiveParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 1, max: 1 },
  inferShape: ([input], p) => imageLike(input!, p.output[0], p.output[1], image(input!).c),
  countParams: () => NO_PARAMS,
  flops: ([input]) => elementsPerSample(input!),
  glyph: "funnel",
  visualizer: "pool-window",
  explain: (p, io) => {
    const { h, w } = image(io.in[0]!);
    return `The ${h}×${w} map is divided into a ${p.output[0]}×${p.output[1]} grid and each cell is averaged, whatever the input size. No weights.`;
  },
  codegen: {
    keras: (p) => `layers.AveragePooling2D()  # adaptive to ${tuple(p.output)}`,
    pytorch: (p) => `nn.AdaptiveAvgPool2d(${tuple(p.output)})`,
  },
});

const upsampleParams = z.object({
  /** Either a factor per axis or an exact output size. */
  scale: pair.optional(),
  size: pair.optional(),
  mode: z.enum(["nearest", "bilinear"]).default("nearest"),
});
type UpsampleParams = z.infer<typeof upsampleParams>;

export const upsample = defineOp<UpsampleParams>({
  op: "upsample",
  family: "reshape",
  title: "Resize (upsample)",
  aliases: {
    keras: ["UpSampling2D"],
    pytorch: ["nn.Upsample"],
    functional: ["F.interpolate", "tf.image.resize"],
  },
  params: upsampleParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {
    keras: { positional: ["size"], keys: { size: "scale", interpolation: "mode" } },
    pytorch: {
      positional: ["size", "scale_factor", "mode"],
      keys: { size: "size", scale_factor: "scale", mode: "mode" },
    },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([input], p) => {
    const { h, w, c } = image(input!);
    if (p.size) return imageLike(input!, p.size[0], p.size[1], c);
    if (p.scale) return imageLike(input!, h * p.scale[0], w * p.scale[1], c);
    throw new ShapeError("upsample needs a scale or a target size");
  },
  countParams: () => NO_PARAMS,
  flops: (_inputs, out) => elementsPerSample(out),
  glyph: "ribbon",
  visualizer: "none",
  explain: (p, io) => {
    const i = image(io.in[0]!);
    const o = image(io.out, "output");
    return `The ${i.h}×${i.w} map is resized to ${o.h}×${o.w} by ${p.mode === "bilinear" ? "blending the four nearest values" : "repeating values"}. Channels and weights are unchanged.`;
  },
  codegen: {
    keras: (p, io) => {
      const o = image(io.out, "output");
      return p.scale
        ? `layers.UpSampling2D(size=${tuple(p.scale)}, interpolation="${p.mode}")`
        : `layers.Resizing(${o.h}, ${o.w}, interpolation="${p.mode}")`;
    },
    pytorch: (p) =>
      p.size
        ? `nn.Upsample(size=${tuple(p.size)}, mode="${p.mode}")`
        : `nn.Upsample(scale_factor=${tuple(p.scale ?? [2, 2])}, mode="${p.mode}")`,
  },
});

const reduceParams = z.object({
  /** One axis or several, written as in the framework (batch is axis 0, negatives count from the end). */
  axis: z.preprocess(
    (value) => (typeof value === "number" ? [value] : value),
    z.array(z.number().int()).min(1),
  ),
  keepDims: z.boolean().default(false),
});
type ReduceParams = z.infer<typeof reduceParams>;

function reduce(kind: "mean" | "max"): AnyOpSpec {
  return defineOp<ReduceParams>({
    op: kind === "mean" ? "reduce_mean" : "reduce_max",
    family: "merge",
    title: kind === "mean" ? "Mean over an axis" : "Max over an axis",
    aliases: {
      functional:
        kind === "mean" ? ["torch.mean", "tf.reduce_mean"] : ["torch.amax", "tf.reduce_max"],
    },
    params: reduceParams,
    defaults: { keras: {}, pytorch: {} },
    signatures: {},
    arity: { min: 1, max: 1 },
    inferShape: ([input], p, ctx) => {
      const shape = input!;
      const axes = new Set(p.axis.map((axis) => resolveAxis(axis, shape.dims.length)));
      if (axes.has(0)) throw new ShapeError("reducing over the batch axis is not supported");
      const dims: Dim[] = [];
      shape.dims.forEach((dim, index) => {
        if (!axes.has(index)) dims.push(dim);
        else if (p.keepDims) dims.push(1);
      });
      return {
        dims,
        layout: p.keepDims ? shape.layout : layoutForRank(dims.length, ctx.framework),
      };
    },
    countParams: () => NO_PARAMS,
    flops: ([input]) => elementsPerSample(input!),
    glyph: "merge",
    visualizer: "none",
    explain: (p, io) =>
      `Takes the ${kind === "mean" ? "average" : "largest value"} along axis ${p.axis.join(", ")}, turning ${shapeText(io.in[0]!)} into ${shapeText(io.out)}. No weights.`,
    codegen: {
      keras: (p) =>
        `layers.Lambda(lambda t: keras.ops.${kind}(t, axis=${tuple(p.axis)}, keepdims=${p.keepDims ? "True" : "False"}))`,
      pytorch: (p) =>
        `# x.${kind === "mean" ? "mean" : "amax"}(dim=${tuple(p.axis)}, keepdim=${p.keepDims ? "True" : "False"})`,
    },
  });
}

// ── Attention modules ─────────────────────────────────────────────────

const gateParams = z.object({
  /** The hidden layer has channels / reduction units, but never fewer than minHidden. */
  reduction: posInt.default(16),
  minHidden: posInt.default(1),
  bias: z.boolean().default(false),
});
type GateParams = z.infer<typeof gateParams>;

const hiddenUnits = (channels: number, p: GateParams) =>
  Math.max(Math.floor(channels / p.reduction), p.minHidden);

function channelGate(op: "squeeze_excite" | "channel_attention"): AnyOpSpec {
  const cbam = op === "channel_attention";
  return defineOp<GateParams>({
    op,
    family: "attention",
    title: cbam ? "Channel Attention (CBAM)" : "Squeeze-and-Excite",
    aliases: {},
    params: gateParams,
    // Squeeze-excite layers normally have biases; CBAM's shared MLP does not.
    defaults: cbam
      ? { keras: {}, pytorch: {} }
      : { keras: { bias: true }, pytorch: { bias: true } },
    signatures: {},
    arity: { min: 1, max: 1 },
    inferShape: ([input]) => {
      image(input!);
      return input!;
    },
    countParams: ([input], p) => {
      const { c } = image(input!);
      const hidden = hiddenUnits(c, p);
      return { trainable: c * hidden + hidden * c + (p.bias ? hidden + c : 0), nonTrainable: 0 };
    },
    flops: ([input], _out, p) => {
      const { c } = image(input!);
      return 2 * 2 * c * hiddenUnits(c, p) * (cbam ? 2 : 1) + 2 * elementsPerSample(input!);
    },
    glyph: "ring",
    visualizer: "none",
    explain: (p, io) => {
      const { c } = image(io.in[0]!);
      const hidden = hiddenUnits(c, p);
      return cbam
        ? `Each of the ${c} channels is summarized by its average and its maximum. A small shared network (${c} → ${hidden} → ${c}) turns both summaries into one weight per channel, and the channels are scaled by those weights. It answers "which features matter here?".`
        : `Each of the ${c} channels is averaged to one number, a small network (${c} → ${hidden} → ${c}) turns those into one weight per channel, and the channels are scaled by those weights.`;
    },
    codegen: {
      keras: () =>
        `# ${cbam ? "CBAM channel attention" : "squeeze-and-excite"}: write as a small custom Layer`,
      pytorch: (p, io) => {
        const { c } = image(io.in[0]!);
        return cbam
          ? `ChannelAttention(${c}, reduction=${p.reduction})`
          : `torchvision.ops.SqueezeExcitation(${c}, ${hiddenUnits(c, p)})`;
      },
    },
  });
}

const spatialParams = z.object({ kernel: posInt.default(7), bias: z.boolean().default(false) });
type SpatialParams = z.infer<typeof spatialParams>;

export const spatialAttention = defineOp<SpatialParams>({
  op: "spatial_attention",
  family: "attention",
  title: "Spatial Attention (CBAM)",
  aliases: {},
  params: spatialParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 1, max: 1 },
  inferShape: ([input]) => {
    image(input!);
    return input!;
  },
  countParams: (_inputs, p) => ({
    trainable: p.kernel * p.kernel * 2 + (p.bias ? 1 : 0),
    nonTrainable: 0,
  }),
  flops: ([input], _out, p) => {
    const { h, w } = image(input!);
    return 2 * h * w * p.kernel * p.kernel * 2 + 2 * elementsPerSample(input!);
  },
  glyph: "ring",
  visualizer: "none",
  explain: (p, io) => {
    const { h, w } = image(io.in[0]!);
    return `At every position the channels are summarized by their average and their maximum. A ${p.kernel}×${p.kernel} convolution turns those two maps into one ${h}×${w} map of weights, and every position is scaled by its weight. It answers "where should the model look?".`;
  },
  codegen: {
    keras: () => "# CBAM spatial attention: write as a small custom Layer",
    pytorch: (p) => `SpatialAttention(k=${p.kernel})`,
  },
});

// ── Recurrent layers ──────────────────────────────────────────────────

const recurrentParams = z.object({
  units: posInt,
  /** True: one output per time step. False: only the last one. */
  returnSequences: z.boolean().default(false),
  bidirectional: z.boolean().default(false),
  bias: z.boolean().default(true),
});
type RecurrentParams = z.infer<typeof recurrentParams>;

function recurrent(
  op: "rnn" | "lstm" | "gru",
  title: string,
  gates: number,
  what: string,
): AnyOpSpec {
  const kerasName = op === "rnn" ? "SimpleRNN" : op.toUpperCase();
  const torchName = op === "rnn" ? "nn.RNN" : `nn.${op.toUpperCase()}`;
  return defineOp<RecurrentParams>({
    op,
    family: "recurrent",
    title,
    aliases: { keras: [kerasName], pytorch: [torchName] },
    params: recurrentParams,
    // nn.LSTM returns the whole sequence; Keras returns the last step unless asked.
    defaults: { keras: { returnSequences: false }, pytorch: { returnSequences: true } },
    signatures: {
      keras: {
        positional: ["units"],
        keys: { units: "units", return_sequences: "returnSequences", use_bias: "bias" },
      },
      pytorch: {
        positional: ["input_size", "hidden_size", "num_layers", "bias"],
        keys: { hidden_size: "units", bias: "bias", bidirectional: "bidirectional" },
      },
    },
    arity: { min: 1, max: 1 },
    inferShape: ([input], p) => {
      const shape = input!;
      if (shape.dims.length !== 3)
        throw new ShapeError(
          `a recurrent layer needs a sequence (B, steps, size), got ${shapeText(shape)}`,
        );
      lastDim(shape);
      const out = p.units * (p.bidirectional ? 2 : 1);
      const batch = shape.dims[0] ?? BATCH;
      return p.returnSequences
        ? { dims: [batch, shape.dims[1]!, out], layout: "NLC" }
        : { dims: [batch, out], layout: "NC" };
    },
    countParams: ([input], p, ctx) => {
      const size = lastDim(input!);
      // PyTorch keeps two bias vectors per gate; Keras keeps one (two for GRU's default reset_after).
      const biases = !p.bias ? 0 : ctx.framework === "pytorch" || op === "gru" ? 2 : 1;
      const perDirection = gates * (p.units * (size + p.units) + biases * p.units);
      return { trainable: perDirection * (p.bidirectional ? 2 : 1), nonTrainable: 0 };
    },
    flops: ([input], _out, p) => {
      const steps = known(input!.dims[1], "steps");
      return 2 * steps * gates * p.units * (lastDim(input!) + p.units) * (p.bidirectional ? 2 : 1);
    },
    glyph: "ring",
    visualizer: "none",
    explain: (p, io) => {
      const steps = io.in[0]!.dims[1];
      return `Reads the sequence one step at a time (${typeof steps === "number" ? steps : "T"} steps), carrying a memory of ${p.units} numbers from step to step. ${what} It returns ${p.returnSequences ? "its memory after every step" : "only its final memory"}${p.bidirectional ? ", reading the sequence in both directions" : ""}.`;
    },
    codegen: {
      keras: (p) => {
        const layer = `layers.${kerasName}(${p.units}${p.returnSequences ? ", return_sequences=True" : ""})`;
        return p.bidirectional ? `layers.Bidirectional(${layer})` : layer;
      },
      pytorch: (p, io) =>
        `${torchName}(${lastDim(io.in[0]!)}, ${p.units}, batch_first=True${p.bidirectional ? ", bidirectional=True" : ""})`,
    },
  });
}

// ── Detection ─────────────────────────────────────────────────────────

const anchorParams = z.object({
  /** Boxes proposed at every position of the feature map (scales × aspect ratios). */
  perLocation: posInt.default(9),
  /** Pixels of the input image per feature-map cell. Only used for the explanation. */
  stride: posInt.optional(),
});
type AnchorParams = z.infer<typeof anchorParams>;

export const anchorGenerator = defineOp<AnchorParams>({
  op: "anchor_generator",
  family: "detection",
  title: "Anchor Boxes",
  aliases: { pytorch: ["AnchorGenerator"] },
  params: anchorParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 1, max: 1 },
  inferShape: ([input], p) => {
    const { h, w } = image(input!, "the feature map");
    return { dims: [input!.dims[0] ?? BATCH, h * w * p.perLocation, 4], layout: "NLC" };
  },
  countParams: () => NO_PARAMS,
  flops: () => 0,
  glyph: "box",
  visualizer: "roi-grid",
  explain: (p, io) => {
    const { h, w } = image(io.in[0]!);
    const stride = p.stride ? ` (one every ${p.stride} pixels of the image)` : "";
    return `${p.perLocation} reference boxes of different sizes and shapes are placed at each of the ${h}×${w} positions${stride}: ${(h * w * p.perLocation).toLocaleString("en-US")} boxes. The detector predicts, for each one, whether it holds an object and how to adjust it. Nothing is learned here.`;
  },
  codegen: {
    keras: (p) => `# anchor boxes: ${p.perLocation} per feature-map position`,
    pytorch: (p) => `# AnchorGenerator: ${p.perLocation} boxes per position`,
  },
});

export const extraOps = [
  groupnorm,
  ...moreActivations,
  globalMaxPool,
  adaptiveAvgPool2d,
  upsample,
  reduce("mean"),
  reduce("max"),
  channelGate("squeeze_excite"),
  channelGate("channel_attention"),
  spatialAttention,
  recurrent("rnn", "Simple RNN", 1, "Each step mixes the new input with the old memory."),
  recurrent(
    "lstm",
    "LSTM",
    4,
    "Four gates decide what to forget, what to store and what to output, which lets it remember over long sequences.",
  ),
  recurrent(
    "gru",
    "GRU",
    3,
    "Three gates decide how much of the old memory to keep and how much to overwrite.",
  ),
  anchorGenerator,
];
