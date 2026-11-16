/**
 * 1D convolution and pooling for sequences, and the detection steps that come
 * after a detector's heads: RoI align, box decoding and non-maximum suppression.
 */
import { z } from "zod";
import {
  BATCH,
  elementsPerSample,
  image,
  isKnown,
  known,
  ShapeError,
  type Dim,
  type TensorShape,
} from "../shapes/tensor";
import { windowOut } from "./conv";
import {
  activation,
  defineOp,
  kerasActivation,
  kerasArgs,
  NO_PARAMS,
  pair,
  posInt,
  type ShapeCtx,
} from "./types";

// ── Sequences ─────────────────────────────────────────────────────────

/** `3` or `[3]` → `3`. */
const one = z.preprocess((value) => (Array.isArray(value) ? value[0] : value), posInt);

/** "valid" / "same", or explicit padding per side. */
const padding1 = z.preprocess(
  (value) => {
    if (typeof value === "string") return value.toLowerCase();
    return Array.isArray(value) ? value[0] : value;
  },
  z.union([z.enum(["valid", "same"]), z.number().int().nonnegative()]),
);

type SeqView = { length: number; channels: number; channelFirst: boolean };

/**
 * Length and channels of a (batch, length, channels) tensor. PyTorch's 1D layers
 * read their input as (batch, channels, length), whatever produced it.
 */
function sequence(shape: TensorShape, channelFirst: boolean, what = "input"): SeqView {
  if (shape.dims.length !== 3) {
    throw new ShapeError(`${what} must be a sequence (batch, length, channels)`);
  }
  return {
    length: known(shape.dims[channelFirst ? 2 : 1], "sequence length"),
    channels: known(shape.dims[channelFirst ? 1 : 2], "channels"),
    channelFirst,
  };
}

const readsChannelFirst = (shape: TensorShape, ctx: ShapeCtx) =>
  shape.layout === "NCL" || ctx.framework === "pytorch";

function sequenceLike(like: TensorShape, view: SeqView, length: number, channels: number) {
  const batch = like.dims[0] ?? BATCH;
  return view.channelFirst
    ? ({ dims: [batch, channels, length], layout: "NCL" } as TensorShape)
    : ({ dims: [batch, length, channels], layout: "NLC" } as TensorShape);
}

const conv1dParams = z.object({
  filters: posInt,
  kernel: one,
  stride: one.default(1),
  padding: padding1.default("valid"),
  dilation: one.default(1),
  groups: posInt.default(1),
  bias: z.boolean().default(true),
  activation: activation.default("linear"),
});
type Conv1dParams = z.infer<typeof conv1dParams>;

export const conv1d = defineOp<Conv1dParams>({
  op: "conv1d",
  family: "conv",
  title: "1D Convolution",
  aliases: {
    keras: ["Conv1D", "Convolution1D"],
    pytorch: ["nn.Conv1d"],
    functional: ["F.conv1d"],
  },
  params: conv1dParams,
  defaults: { keras: { padding: "valid" }, pytorch: { padding: 0 } },
  signatures: {
    keras: {
      positional: ["filters", "kernel_size", "strides", "padding"],
      keys: {
        filters: "filters",
        kernel_size: "kernel",
        strides: "stride",
        padding: "padding",
        dilation_rate: "dilation",
        groups: "groups",
        use_bias: "bias",
        activation: "activation",
      },
    },
    pytorch: {
      positional: [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
      ],
      keys: {
        out_channels: "filters",
        kernel_size: "kernel",
        stride: "stride",
        padding: "padding",
        dilation: "dilation",
        groups: "groups",
        bias: "bias",
      },
    },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([input], p, ctx) => {
    const view = sequence(input!, readsChannelFirst(input!, ctx));
    if (view.channels % p.groups !== 0 || p.filters % p.groups !== 0) {
      throw new ShapeError(
        `groups=${p.groups} must divide both ${view.channels} input and ${p.filters} output channels`,
      );
    }
    const length = windowOut(view.length, p.kernel, p.stride, p.padding, p.dilation, ctx);
    return sequenceLike(input!, view, length, p.filters);
  },
  countParams: ([input], p, ctx) => {
    const { channels } = sequence(input!, readsChannelFirst(input!, ctx));
    const weights = p.kernel * (channels / p.groups) * p.filters;
    return { trainable: weights + (p.bias ? p.filters : 0), nonTrainable: 0 };
  },
  flops: ([input], out, p) => {
    const first = out.layout === "NCL";
    const { channels } = sequence(input!, first);
    const { length } = sequence(out, first, "output");
    return 2 * length * p.kernel * (channels / p.groups) * p.filters;
  },
  glyph: "volume",
  visualizer: "none",
  explain: (p, io) => {
    const first = io.out.layout === "NCL";
    const { channels, length: before } = sequence(io.in[0]!, first);
    const { length } = sequence(io.out, first, "output");
    const act = p.activation === "linear" ? "" : `, then ${p.activation}`;
    const order = first ? " PyTorch reads the input as (batch, channels, length)." : "";
    return `${p.filters} filters, each ${p.kernel} steps wide over ${channels / p.groups} channels, slide along the sequence with stride ${p.stride}. ${before} steps become ${length}${act}.${order}`;
  },
  codegen: {
    keras: (p) =>
      `layers.Conv1D(${kerasArgs([
        String(p.filters),
        String(p.kernel),
        `strides=${p.stride}`,
        typeof p.padding === "string" ? `padding="${p.padding}"` : null,
        p.dilation > 1 && `dilation_rate=${p.dilation}`,
        p.groups > 1 && `groups=${p.groups}`,
        !p.bias && "use_bias=False",
        kerasActivation(p.activation),
      ])})`,
    pytorch: (p, io) => {
      const { channels } = sequence(io.in[0]!, true);
      const pad = p.padding === "valid" ? "0" : p.padding === "same" ? '"same"' : p.padding;
      return `nn.Conv1d(${channels}, ${p.filters}, kernel_size=${p.kernel}, stride=${p.stride}, padding=${pad}${p.dilation > 1 ? `, dilation=${p.dilation}` : ""}${p.groups > 1 ? `, groups=${p.groups}` : ""}${p.bias ? "" : ", bias=False"})`;
    },
  },
});

const pool1dParams = z.object({
  pool: one,
  /** Missing stride means "same as the pool size" (the default in both frameworks). */
  stride: one.optional(),
  padding: padding1.default("valid"),
});
type Pool1dParams = z.infer<typeof pool1dParams>;

export const maxpool1d = defineOp<Pool1dParams>({
  op: "maxpool1d",
  family: "pool",
  title: "1D Max Pooling",
  aliases: { keras: ["MaxPooling1D", "MaxPool1D"], pytorch: ["nn.MaxPool1d"] },
  params: pool1dParams,
  defaults: { keras: { padding: "valid" }, pytorch: { padding: 0 } },
  signatures: {
    keras: {
      positional: ["pool_size", "strides", "padding"],
      keys: { pool_size: "pool", strides: "stride", padding: "padding" },
    },
    pytorch: {
      positional: ["kernel_size", "stride", "padding"],
      keys: { kernel_size: "pool", stride: "stride", padding: "padding" },
    },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([input], p, ctx) => {
    const view = sequence(input!, readsChannelFirst(input!, ctx));
    const length = windowOut(view.length, p.pool, p.stride ?? p.pool, p.padding, 1, ctx);
    return sequenceLike(input!, view, length, view.channels);
  },
  countParams: () => NO_PARAMS,
  flops: (_inputs, out, p) => elementsPerSample(out) * p.pool,
  glyph: "funnel",
  visualizer: "none",
  explain: (p, io) => {
    const first = io.out.layout === "NCL";
    const { length: before } = sequence(io.in[0]!, first);
    const { length } = sequence(io.out, first, "output");
    return `A window of ${p.pool} steps moves along the sequence with stride ${p.stride ?? p.pool} and keeps the largest value of each channel, shrinking ${before} steps to ${length}. No weights.`;
  },
  codegen: {
    keras: (p) =>
      `layers.MaxPooling1D(${kerasArgs([
        `pool_size=${p.pool}`,
        `strides=${p.stride ?? p.pool}`,
        typeof p.padding === "string" ? `padding="${p.padding}"` : null,
      ])})`,
    pytorch: (p) =>
      `nn.MaxPool1d(kernel_size=${p.pool}, stride=${p.stride ?? p.pool}${typeof p.padding === "number" && p.padding > 0 ? `, padding=${p.padding}` : ""})`,
  },
});

// ── Detection ─────────────────────────────────────────────────────────

/** Checks a (batch, N, 4) box tensor and returns N. */
function boxCount(shape: TensorShape | undefined, what: string): Dim {
  if (!shape || shape.dims.length !== 3 || shape.dims[2] !== 4) {
    throw new ShapeError(`${what} must be boxes with shape (B, N, 4)`);
  }
  return shape.dims[1]!;
}

const sameCount = (a: Dim, b: Dim) => !isKnown(a) || !isKnown(b) || a === b;

const roiAlignParams = z.object({
  pooled: pair,
  /** Sample points per output cell and axis. */
  samplingRatio: posInt.default(2),
});
type RoiAlignParams = z.infer<typeof roiAlignParams>;

export const roiAlign = defineOp<RoiAlignParams>({
  op: "roi_align",
  family: "detection",
  title: "RoI Align",
  aliases: {
    pytorch: ["RoIAlign", "torchvision.ops.RoIAlign"],
    functional: ["roi_align", "torchvision.ops.roi_align"],
  },
  params: roiAlignParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {
    pytorch: {
      positional: ["output_size", "spatial_scale", "sampling_ratio"],
      keys: { output_size: "pooled", sampling_ratio: "samplingRatio" },
    },
  },
  arity: { min: 2, max: 2 },
  // Inputs: the shared feature map, then the boxes (B, R, 4).
  inferShape: ([features, boxes], p) => {
    const { c } = image(features!, "the feature map");
    const rois = boxCount(boxes, "the second input");
    return {
      dims: [features!.dims[0] ?? BATCH, rois, p.pooled[0], p.pooled[1], c],
      layout: "any",
    };
  },
  countParams: () => NO_PARAMS,
  // Four neighbours are blended for every sample point.
  flops: (_inputs, out, p) => elementsPerSample(out) * 4 * p.samplingRatio ** 2,
  glyph: "box",
  visualizer: "roi-grid",
  explain: (p, io) => {
    const { h, w, c } = image(io.in[0]!);
    return `For every proposed box, the matching region of the ${h}×${w}×${c} feature map is resampled to a fixed ${p.pooled[0]}×${p.pooled[1]} grid. Unlike RoI pooling, positions are not rounded: each cell averages ${p.samplingRatio}×${p.samplingRatio} points read by bilinear interpolation, which keeps small objects aligned. No weights.`;
  },
  codegen: {
    keras: (p) => `# RoI align to ${p.pooled[0]}×${p.pooled[1]} (tf.image.crop_and_resize)`,
    pytorch: (p) =>
      `torchvision.ops.RoIAlign(output_size=(${p.pooled.join(", ")}), spatial_scale=1.0, sampling_ratio=${p.samplingRatio})`,
  },
});

const boxDecodeParams = z.object({
  /** Keep decoded boxes inside the image. */
  clip: z.boolean().default(true),
});
type BoxDecodeParams = z.infer<typeof boxDecodeParams>;

export const boxDecode = defineOp<BoxDecodeParams>({
  op: "box_decode",
  family: "detection",
  title: "Box Decoding",
  aliases: { functional: ["decode_boxes", "box_coder.decode", "apply_deltas"] },
  params: boxDecodeParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 2, max: 2 },
  // Inputs: the predicted offsets (B, N, 4), then the anchors or proposals (B, N, 4).
  inferShape: ([deltas, anchors]) => {
    const count = boxCount(deltas, "the first input (predicted offsets)");
    const anchorCount = boxCount(anchors, "the second input (anchors)");
    if (!sameCount(count, anchorCount)) {
      throw new ShapeError(
        `there are ${String(count)} predicted offsets but ${String(anchorCount)} anchors`,
      );
    }
    return { dims: [deltas!.dims[0] ?? BATCH, count, 4], layout: "NLC" };
  },
  countParams: () => NO_PARAMS,
  flops: (_inputs, out) => elementsPerSample(out) * 4,
  glyph: "box",
  visualizer: "roi-grid",
  explain: (p, io) => {
    const count = io.out.dims[1]!;
    const boxes = isKnown(count) ? count.toLocaleString("en-US") : "all";
    return `The network does not predict boxes directly: it predicts, for each of the ${boxes} anchors, how far to shift its centre and how much to stretch its width and height. This step applies those four offsets to each anchor to get real box corners${p.clip ? ", then clips them to the image" : ""}. Nothing is learned here.`;
  },
  codegen: {
    keras: () => "# decode: anchors + predicted offsets → boxes (x1, y1, x2, y2)",
    pytorch: () => "# box_coder.decode(deltas, anchors) → boxes (x1, y1, x2, y2)",
  },
});

const nmsParams = z.object({
  iouThreshold: z.number().min(0).max(1).default(0.5),
  scoreThreshold: z.number().min(0).max(1).default(0.05),
  maxOutput: posInt.default(100),
});
type NmsParams = z.infer<typeof nmsParams>;

export const nms = defineOp<NmsParams>({
  op: "nms",
  family: "detection",
  title: "Non-Maximum Suppression",
  aliases: {
    functional: [
      "nms",
      "batched_nms",
      "torchvision.ops.nms",
      "torchvision.ops.batched_nms",
      "tf.image.non_max_suppression",
      "tf.image.combined_non_max_suppression",
    ],
  },
  params: nmsParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 2, max: 2 },
  // Inputs: boxes (B, N, 4), then scores (B, N) or (B, N, classes).
  inferShape: ([boxes, scores]) => {
    const count = boxCount(boxes, "the first input");
    if (!scores || scores.dims.length < 2 || scores.dims.length > 3) {
      throw new ShapeError("the second input must be scores with shape (B, N) or (B, N, classes)");
    }
    if (!sameCount(count, scores.dims[1]!)) {
      throw new ShapeError(`there are ${String(count)} boxes but ${String(scores.dims[1])} scores`);
    }
    // How many boxes survive depends on the picture, so the count is only known at run time.
    const kept: Dim = isKnown(count) && count <= 1 ? count : { sym: "K" };
    return { dims: [boxes!.dims[0] ?? BATCH, kept, 4], layout: "NLC" };
  },
  countParams: () => NO_PARAMS,
  // Sorting plus one overlap test per pair in the worst case; counted as one pass here.
  flops: ([boxes]) => elementsPerSample(boxes!) * 4,
  glyph: "box",
  visualizer: "roi-grid",
  explain: (p) =>
    `Many boxes usually cover the same object. Boxes scoring under ${p.scoreThreshold} are dropped, the rest are taken from best to worst, and a box is thrown away when it overlaps an already kept one by more than ${p.iouThreshold} (intersection over union). At most ${p.maxOutput} boxes remain: K of them, a number that depends on the picture. Nothing is learned here.`,
  codegen: {
    keras: (p) =>
      `tf.image.non_max_suppression(boxes, scores, max_output_size=${p.maxOutput}, iou_threshold=${p.iouThreshold}, score_threshold=${p.scoreThreshold})`,
    pytorch: (p) => `torchvision.ops.nms(boxes, scores, iou_threshold=${p.iouThreshold})`,
  },
});

export const sequenceDetectOps = [conv1d, maxpool1d, roiAlign, boxDecode, nms];
