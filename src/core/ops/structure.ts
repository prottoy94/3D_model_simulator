/** Reshape, merge, detection and declared (custom / backbone) ops. */
import { z } from "zod";
import {
  BATCH,
  dimsEqual,
  dimText,
  elementsPerSample,
  image,
  known,
  layoutForRank,
  resolveAxis,
  ShapeError,
  shapeText,
  type Dim,
  type TensorShape,
} from "../shapes/tensor";
import type { OpKind } from "./kinds";
import { defineOp, NO_PARAMS, pair, tuple, type AnyOpSpec } from "./types";

const product = (dims: Dim[], what: string) =>
  dims.reduce<number>((total, dim) => total * known(dim, what), 1);

// ── Reshape family ────────────────────────────────────────────────────

const flattenParams = z.object({ startAxis: z.number().int().default(1) });

type FlattenParams = z.infer<typeof flattenParams>;

export const flatten = defineOp<FlattenParams>({
  op: "flatten",
  family: "reshape",
  title: "Flatten",
  aliases: { keras: ["Flatten"], pytorch: ["nn.Flatten"], functional: ["torch.flatten"] },
  params: flattenParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {
    keras: { positional: [], keys: {} },
    pytorch: { positional: ["start_dim"], keys: { start_dim: "startAxis" } },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([shape], p, ctx) => {
    const start = resolveAxis(p.startAxis, shape!.dims.length);
    if (start === 0) throw new ShapeError("flatten must keep the batch axis");
    const dims = [
      ...shape!.dims.slice(0, start),
      product(shape!.dims.slice(start), "a flattened dimension"),
    ];
    return { dims, layout: layoutForRank(dims.length, ctx.framework) };
  },
  countParams: () => NO_PARAMS,
  flops: () => 0,
  glyph: "ribbon",
  visualizer: "flatten-unroll",
  explain: (_p, io) =>
    `The ${shapeText(io.in[0]!)} tensor is unrolled into one long row of ${dimText(io.out.dims[io.out.dims.length - 1]!)} numbers. Nothing is computed and no value changes.`,
  codegen: {
    keras: () => "layers.Flatten()",
    pytorch: (p) => `nn.Flatten(${p.startAxis === 1 ? "" : p.startAxis})`,
  },
});

const reshapeParams = z.object({
  /** Target shape without the batch. One entry may be -1 (worked out from the rest). */
  shape: z.array(z.number().int()).min(1),
});

type ReshapeParams = z.infer<typeof reshapeParams>;

export const reshape = defineOp<ReshapeParams>({
  op: "reshape",
  family: "reshape",
  title: "Reshape",
  aliases: { keras: ["Reshape"], functional: ["tf.reshape", "torch.reshape"] },
  params: reshapeParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: { keras: { positional: ["target_shape"], keys: { target_shape: "shape" } } },
  arity: { min: 1, max: 1 },
  inferShape: ([shape], p, ctx) => {
    const total = product(shape!.dims.slice(1), "an input dimension");
    const wild = p.shape.filter((size) => size === -1).length;
    const fixed = p.shape.filter((size) => size !== -1).reduce((a, b) => a * b, 1);
    if (wild > 1 || p.shape.some((size) => size === 0 || size < -1)) {
      throw new ShapeError(`invalid target shape ${tuple(p.shape)}`);
    }
    if (wild === 1 ? total % fixed !== 0 : total !== fixed) {
      throw new ShapeError(`cannot reshape ${total} values into ${tuple(p.shape)}`);
    }
    const dims = [
      shape!.dims[0] ?? BATCH,
      ...p.shape.map((size) => (size === -1 ? total / fixed : size)),
    ];
    return { dims, layout: layoutForRank(dims.length, ctx.framework) };
  },
  countParams: () => NO_PARAMS,
  flops: () => 0,
  glyph: "ribbon",
  visualizer: "flatten-unroll",
  explain: (_p, io) =>
    `The same values are rearranged from ${shapeText(io.in[0]!)} into ${shapeText(io.out)}. Nothing is computed.`,
  codegen: {
    keras: (p) => `layers.Reshape(${tuple(p.shape)})`,
    pytorch: (p) => `# x.reshape(x.shape[0], ${p.shape.join(", ")})`,
  },
});

const sliceParams = z.object({
  axis: z.number().int().default(1),
  /** Pick one position and drop the axis, e.g. the CLS token: axis 1, index 0. */
  index: z.number().int().optional(),
  start: z.number().int().optional(),
  end: z.number().int().optional(),
});

type SliceParams = z.infer<typeof sliceParams>;

export const slice = defineOp<SliceParams>({
  op: "slice",
  family: "reshape",
  title: "Slice",
  aliases: { functional: ["tf.gather", "torch.select"] },
  params: sliceParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 1, max: 1 },
  inferShape: ([shape], p, ctx) => {
    const axis = resolveAxis(p.axis, shape!.dims.length);
    if (axis === 0) throw new ShapeError("slicing the batch axis is not supported");
    const dims = [...shape!.dims];
    if (p.index !== undefined) {
      dims.splice(axis, 1);
      return { dims, layout: layoutForRank(dims.length, ctx.framework) };
    }
    const size = known(dims[axis], "the sliced dimension");
    const clamp = (value: number) => Math.max(0, Math.min(size, value < 0 ? size + value : value));
    const length = clamp(p.end ?? size) - clamp(p.start ?? 0);
    if (length < 1) throw new ShapeError("the slice is empty");
    dims[axis] = length;
    return { dims, layout: shape!.layout };
  },
  countParams: () => NO_PARAMS,
  flops: () => 0,
  glyph: "ribbon",
  visualizer: "none",
  explain: (p) =>
    p.index !== undefined
      ? `Keeps only position ${p.index} along axis ${p.axis} and drops the rest (for example, the class token of a ViT).`
      : `Keeps positions ${p.start ?? 0} to ${p.end ?? "the end"} along axis ${p.axis}.`,
  codegen: {
    keras: (p) => `layers.Lambda(lambda t: t[${sliceText(p)}])`,
    pytorch: (p) => `# x[${sliceText(p)}]`,
  },
});

function sliceText(p: SliceParams): string {
  const lead = Array.from({ length: Math.max(p.axis, 0) }, () => ":");
  return [
    ...lead,
    p.index !== undefined ? String(p.index) : `${p.start ?? ""}:${p.end ?? ""}`,
  ].join(", ");
}

// ── Merge family ──────────────────────────────────────────────────────

/** NumPy-style broadcasting, aligned from the last axis. */
function broadcast(inputs: TensorShape[]): TensorShape {
  if (inputs.length < 2) throw new ShapeError("needs at least two inputs");
  const widest = inputs.reduce((a, b) => (b.dims.length > a.dims.length ? b : a));
  const rank = widest.dims.length;
  const dims: Dim[] = [];
  for (let offset = 1; offset <= rank; offset++) {
    let result: Dim = 1;
    for (const shape of inputs) {
      const dim = shape.dims[shape.dims.length - offset];
      if (dim === undefined || dim === 1) continue;
      if (result === 1) result = dim;
      else if (!dimsEqual(result, dim)) {
        throw new ShapeError(`shapes do not line up: ${inputs.map(shapeText).join(" and ")}`);
      }
    }
    dims.unshift(result);
  }
  return { dims, layout: widest.layout };
}

function elementwise(
  op: OpKind,
  title: string,
  kerasName: string,
  symbol: string,
  explain: string,
): AnyOpSpec {
  return defineOp<Record<string, never>>({
    op,
    family: "merge",
    title,
    aliases: {
      keras: [kerasName],
      functional: [`torch.${op === "add" ? "add" : "mul"}`, `tf.${op}`],
    },
    params: z.object({}),
    defaults: { keras: {}, pytorch: {} },
    signatures: { keras: { positional: [], keys: {} } },
    arity: { min: 2, max: null },
    inferShape: (inputs) => broadcast(inputs),
    countParams: () => NO_PARAMS,
    flops: (_inputs, out) => elementsPerSample(out),
    glyph: "merge",
    visualizer: "merge-streams",
    explain: () => explain,
    codegen: { keras: () => `layers.${kerasName}()`, pytorch: () => `# a ${symbol} b` },
  });
}

const concatParams = z.object({ axis: z.number().int().default(-1) });

type ConcatParams = z.infer<typeof concatParams>;

export const concat = defineOp<ConcatParams>({
  op: "concat",
  family: "merge",
  title: "Concatenate",
  aliases: { keras: ["Concatenate"], functional: ["torch.cat", "tf.concat", "concatenate"] },
  params: concatParams,
  defaults: { keras: { axis: -1 }, pytorch: { axis: 1 } },
  signatures: { keras: { positional: ["axis"], keys: { axis: "axis" } } },
  arity: { min: 2, max: null },
  inferShape: (inputs, p) => {
    const first = inputs[0]!;
    const axis = resolveAxis(p.axis, first.dims.length);
    let total = 0;
    for (const shape of inputs) {
      const sameRest =
        shape.dims.length === first.dims.length &&
        shape.dims.every((dim, i) => i === axis || dimsEqual(dim, first.dims[i]!));
      if (!sameRest) {
        throw new ShapeError(
          `inputs must match on every axis except ${p.axis}: ${inputs.map(shapeText).join(", ")}`,
        );
      }
      total += known(shape.dims[axis], "the joined dimension");
    }
    const dims = [...first.dims];
    dims[axis] = total;
    return { dims, layout: first.layout };
  },
  countParams: () => NO_PARAMS,
  flops: () => 0,
  glyph: "merge",
  visualizer: "merge-streams",
  explain: (p, io) =>
    `The ${io.in.length} inputs are placed side by side along axis ${p.axis}, so that dimension grows to ${dimText(io.out.dims[resolveAxis(p.axis, io.out.dims.length)]!)}. No values change.`,
  codegen: {
    keras: (p) => `layers.Concatenate(axis=${p.axis})`,
    pytorch: (p) => `# torch.cat([...], dim=${p.axis})`,
  },
});

// ── Detection ─────────────────────────────────────────────────────────

const roiPoolParams = z.object({ pooled: pair });

type RoiPoolParams = z.infer<typeof roiPoolParams>;

export const roiPool = defineOp<RoiPoolParams>({
  op: "roi_pool",
  family: "detection",
  title: "RoI Pooling",
  aliases: {
    pytorch: ["RoIPool", "torchvision.ops.RoIPool"],
    functional: ["tf.image.crop_and_resize", "roi_pool"],
  },
  params: roiPoolParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 2, max: 2 },
  // Inputs: the shared feature map, then the boxes (B, R, 4).
  inferShape: ([features, boxes], p) => {
    const { c } = image(features!, "the feature map");
    if (!boxes || boxes.dims.length !== 3 || boxes.dims[2] !== 4) {
      throw new ShapeError("the second input must be boxes with shape (B, R, 4)");
    }
    return {
      dims: [features!.dims[0] ?? BATCH, boxes.dims[1]!, p.pooled[0], p.pooled[1], c],
      layout: "any",
    };
  },
  countParams: () => NO_PARAMS,
  flops: (_inputs, out) => elementsPerSample(out),
  glyph: "box",
  visualizer: "roi-grid",
  explain: (p, io) => {
    const { h, w, c } = image(io.in[0]!);
    return `For every proposed box, the matching region of the shared ${h}×${w}×${c} feature map is cut out and resized to a fixed ${p.pooled[0]}×${p.pooled[1]} grid, so boxes of any size give the same output size. No weights.`;
  },
  codegen: {
    keras: (p) => `ROIPoolingLayer(pooled_size=${p.pooled[0]})  # tf.image.crop_and_resize`,
    pytorch: (p) => `torchvision.ops.RoIPool(output_size=${tuple(p.pooled)}, spatial_scale=1.0)`,
  },
});

// ── Declared ops: the IR states their shape and counts ────────────────

const declaredParams = z.object({
  /** Output shape without the batch. Negative = only known at run time. */
  outputShape: z.array(z.number().int()).optional(),
  paramCount: z.number().int().nonnegative().default(0),
  nonTrainableCount: z.number().int().nonnegative().default(0),
  description: z.string().default(""),
});

type DeclaredParams = z.infer<typeof declaredParams>;

function declaredOp(op: "custom" | "backbone"): AnyOpSpec {
  const isBackbone = op === "backbone";
  return defineOp<DeclaredParams>({
    op,
    family: isBackbone ? "composite" : "custom",
    title: isBackbone ? "Pretrained backbone" : "Custom layer",
    aliases: {},
    params: declaredParams,
    defaults: { keras: {}, pytorch: {} },
    signatures: {},
    arity: { min: 0, max: null },
    declared: true,
    inferShape: (inputs, p, ctx) => {
      if (!p.outputShape) throw new ShapeError("no output shape was declared for this layer");
      const batch = inputs[0]?.dims[0] ?? BATCH;
      const dims: Dim[] = [
        batch,
        ...p.outputShape.map((size): Dim => (size < 0 ? { sym: "N" } : size)),
      ];
      return { dims, layout: layoutForRank(dims.length, ctx.framework) };
    },
    countParams: (_inputs, p) => ({ trainable: p.paramCount, nonTrainable: p.nonTrainableCount }),
    flops: () => 0,
    glyph: isBackbone ? "capsule" : "box",
    visualizer: "none",
    explain: (p, io) =>
      p.description ||
      (isBackbone
        ? `A pretrained network used as a feature extractor. It outputs ${shapeText(io.out)}.`
        : `A layer defined in the notebook. Its shape and counts are declared, not computed.`),
    codegen: {
      keras: () =>
        isBackbone
          ? "# pretrained backbone (see the notebook)"
          : "# custom layer (see the notebook)",
      pytorch: () =>
        isBackbone
          ? "# pretrained backbone (see the notebook)"
          : "# custom layer (see the notebook)",
    },
  });
}

export const structureOps = [
  flatten,
  reshape,
  slice,
  elementwise(
    "add",
    "Add",
    "Add",
    "+",
    "The inputs are added value by value. In a residual connection this lets the block learn a correction to its input instead of a whole new signal.",
  ),
  elementwise(
    "multiply",
    "Multiply",
    "Multiply",
    "*",
    "The inputs are multiplied value by value. With an attention map, this turns features up or down.",
  ),
  concat,
  roiPool,
  declaredOp("custom"),
  declaredOp("backbone"),
];
