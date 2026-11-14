import { z } from "zod";
import {
  BATCH,
  featureDim,
  image,
  imageLike,
  isImage,
  ShapeError,
  type TensorShape,
} from "../shapes/tensor";
import { padAt, windowOut } from "./conv";
import {
  defineOp,
  kerasArgs,
  NO_PARAMS,
  padding,
  pair,
  times,
  tuple,
  type AnyOpSpec,
} from "./types";

const poolParams = z.object({
  pool: pair,
  /** Missing stride means "same as the pool size" (the default in both frameworks). */
  stride: pair.optional(),
  padding: padding.default("valid"),
});

type PoolParams = z.infer<typeof poolParams>;

function pool2d(kind: "max" | "avg"): AnyOpSpec {
  const kerasName = kind === "max" ? "MaxPooling2D" : "AveragePooling2D";
  const torchName = kind === "max" ? "nn.MaxPool2d" : "nn.AvgPool2d";
  return defineOp<PoolParams>({
    op: kind === "max" ? "maxpool2d" : "avgpool2d",
    family: "pool",
    title: kind === "max" ? "Max Pooling" : "Average Pooling",
    aliases: {
      keras: kind === "max" ? ["MaxPooling2D", "MaxPool2D"] : ["AveragePooling2D", "AvgPool2D"],
      pytorch: [torchName],
      functional: kind === "max" ? ["F.max_pool2d"] : ["F.avg_pool2d"],
    },
    params: poolParams,
    defaults: { keras: { pool: [2, 2], padding: "valid" }, pytorch: { padding: [0, 0] } },
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
      const { h, w, c } = image(input!);
      const stride = p.stride ?? p.pool;
      return imageLike(
        input!,
        windowOut(h, p.pool[0], stride[0], padAt(p.padding, 0), 1, ctx),
        windowOut(w, p.pool[1], stride[1], padAt(p.padding, 1), 1, ctx),
        c,
      );
    },
    countParams: () => NO_PARAMS,
    flops: (_inputs, out, p) => {
      const o = image(out, "output");
      return o.h * o.w * o.c * p.pool[0] * p.pool[1];
    },
    glyph: "funnel",
    visualizer: "pool-window",
    explain: (p, io) => {
      const i = image(io.in[0]!);
      const o = image(io.out, "output");
      const what = kind === "max" ? "keeps the largest value" : "takes the average";
      return `A ${times(p.pool)} window moves with stride ${times(p.stride ?? p.pool)} and ${what} in each position, shrinking ${i.h}×${i.w} to ${o.h}×${o.w}. Channels are untouched.`;
    },
    codegen: {
      keras: (p) =>
        `layers.${kerasName}(${kerasArgs([
          `pool_size=${tuple(p.pool)}`,
          p.stride && `strides=${tuple(p.stride)}`,
          p.padding === "same" && 'padding="same"',
        ])})`,
      pytorch: (p) =>
        `${torchName}(kernel_size=${tuple(p.pool)}, stride=${tuple(p.stride ?? p.pool)}${typeof p.padding === "string" ? "" : `, padding=${tuple(p.padding)}`})`,
    },
  });
}

const globalPoolParams = z.object({ keepDims: z.boolean().default(false) });

type GlobalPoolParams = z.infer<typeof globalPoolParams>;

export const globalAvgPool = defineOp<GlobalPoolParams>({
  op: "global_avg_pool",
  family: "pool",
  title: "Global Average Pooling",
  aliases: {
    keras: [
      "GlobalAveragePooling2D",
      "GlobalAveragePooling1D",
      "GlobalAvgPool2D",
      "GlobalAvgPool1D",
    ],
    pytorch: ["nn.AdaptiveAvgPool2d"],
    functional: ["F.adaptive_avg_pool2d"],
  },
  params: globalPoolParams,
  // nn.AdaptiveAvgPool2d(1) keeps the two spatial axes with size 1.
  defaults: { keras: {}, pytorch: { keepDims: true } },
  signatures: {
    keras: { positional: [], keys: { keepdims: "keepDims" } },
    pytorch: { positional: ["output_size"], keys: {} },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([input], p): TensorShape => {
    const shape = input!;
    const batch = shape.dims[0] ?? BATCH;
    if (isImage(shape)) {
      const { c } = image(shape);
      return p.keepDims ? imageLike(shape, 1, 1, c) : { dims: [batch, c], layout: "NC" };
    }
    if (shape.dims.length === 3) {
      const c = featureDim(shape);
      if (!p.keepDims) return { dims: [batch, c], layout: "NC" };
      return shape.layout === "NCL"
        ? { dims: [batch, c, 1], layout: "NCL" }
        : { dims: [batch, 1, c], layout: "NLC" };
    }
    throw new ShapeError("global pooling needs an image or a sequence");
  },
  countParams: () => NO_PARAMS,
  flops: ([input]) =>
    input!.dims.slice(1).reduce<number>((n, d) => n * (typeof d === "number" ? d : 1), 1),
  glyph: "funnel",
  visualizer: "global-pool",
  explain: (_p, io) => {
    const shape = io.in[0]!;
    const c = featureDim(shape);
    return isImage(shape)
      ? `Each of the ${c} feature maps is averaged down to a single number, giving one vector of ${c} values. No weights.`
      : `The sequence is averaged over its positions, giving one vector of ${c} values. No weights.`;
  },
  codegen: {
    keras: (_p, io) =>
      isImage(io.in[0]!) ? "layers.GlobalAveragePooling2D()" : "layers.GlobalAveragePooling1D()",
    pytorch: (p) =>
      p.keepDims
        ? "nn.AdaptiveAvgPool2d(1)"
        : "nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())",
  },
});

export const poolOps = [pool2d("max"), pool2d("avg"), globalAvgPool];
