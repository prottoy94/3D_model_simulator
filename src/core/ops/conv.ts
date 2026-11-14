import { z } from "zod";
import { image, imageLike, ShapeError, type TensorShape } from "../shapes/tensor";
import {
  activation,
  defineOp,
  kerasActivation,
  kerasArgs,
  padding,
  pair,
  posInt,
  times,
  tuple,
  type Padding,
  type ShapeCtx,
} from "./types";

/**
 * Output size of a sliding window along one axis (doc 03 §4.3).
 * Keras "same" keeps ceil(in / stride); explicit padding uses the PyTorch formula.
 */
export function windowOut(
  size: number,
  kernel: number,
  stride: number,
  pad: "valid" | "same" | number,
  dilation: number,
  ctx: ShapeCtx,
): number {
  if (pad === "same") {
    if (ctx.framework === "pytorch" && stride !== 1) {
      throw new ShapeError('PyTorch only allows padding="same" with stride 1');
    }
    return Math.ceil(size / stride);
  }
  const p = pad === "valid" ? 0 : pad;
  const out = Math.floor((size + 2 * p - dilation * (kernel - 1) - 1) / stride) + 1;
  if (out < 1) {
    throw new ShapeError(`a window of ${kernel} does not fit an input of size ${size}`);
  }
  return out;
}

export function padAt(pad: Padding, axis: 0 | 1): "valid" | "same" | number {
  return typeof pad === "string" ? pad : pad[axis];
}

const pytorchPadding = (pad: Padding) =>
  pad === "valid" ? "0" : pad === "same" ? '"same"' : tuple(pad);

const conv2dParams = z.object({
  filters: posInt,
  kernel: pair,
  stride: pair.default([1, 1]),
  padding: padding.default("valid"),
  dilation: pair.default([1, 1]),
  groups: posInt.default(1),
  bias: z.boolean().default(true),
  activation: activation.default("linear"),
});

type Conv2dParams = z.infer<typeof conv2dParams>;

function convOutput(input: TensorShape, p: Conv2dParams, ctx: ShapeCtx): TensorShape {
  const { h, w, c } = image(input);
  if (c % p.groups !== 0 || p.filters % p.groups !== 0) {
    throw new ShapeError(
      `groups=${p.groups} must divide both ${c} input and ${p.filters} output channels`,
    );
  }
  return imageLike(
    input,
    windowOut(h, p.kernel[0], p.stride[0], padAt(p.padding, 0), p.dilation[0], ctx),
    windowOut(w, p.kernel[1], p.stride[1], padAt(p.padding, 1), p.dilation[1], ctx),
    p.filters,
  );
}

export const conv2d = defineOp<Conv2dParams>({
  op: "conv2d",
  family: "conv",
  title: "2D Convolution",
  aliases: {
    keras: ["Conv2D", "Convolution2D"],
    pytorch: ["nn.Conv2d"],
    functional: ["F.conv2d", "tf.nn.conv2d"],
  },
  params: conv2dParams,
  defaults: { keras: { padding: "valid" }, pytorch: { padding: [0, 0] } },
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
  inferShape: ([input], p, ctx) => convOutput(input!, p, ctx),
  countParams: ([input], p) => {
    const { c } = image(input!);
    const weights = p.kernel[0] * p.kernel[1] * (c / p.groups) * p.filters;
    return { trainable: weights + (p.bias ? p.filters : 0), nonTrainable: 0 };
  },
  flops: ([input], out, p) => {
    const { c } = image(input!);
    const o = image(out, "output");
    return 2 * o.h * o.w * p.kernel[0] * p.kernel[1] * (c / p.groups) * p.filters;
  },
  glyph: "volume",
  visualizer: "conv-sweep",
  explain: (p, io) => {
    const { c } = image(io.in[0]!);
    const o = image(io.out, "output");
    const act = p.activation === "linear" ? "" : `, then ${p.activation}`;
    return `${p.filters} filters of ${times(p.kernel)}×${c / p.groups} slide over the input with stride ${times(p.stride)}. Each filter produces one ${o.h}×${o.w} feature map${act}.`;
  },
  codegen: {
    keras: (p) =>
      `layers.Conv2D(${kerasArgs([
        String(p.filters),
        tuple(p.kernel),
        `strides=${tuple(p.stride)}`,
        typeof p.padding === "string" ? `padding="${p.padding}"` : null,
        p.groups > 1 && `groups=${p.groups}`,
        !p.bias && "use_bias=False",
        kerasActivation(p.activation),
      ])})`,
    pytorch: (p, io) =>
      `nn.Conv2d(${image(io.in[0]!).c}, ${p.filters}, kernel_size=${tuple(p.kernel)}, stride=${tuple(p.stride)}, padding=${pytorchPadding(p.padding)}${p.groups > 1 ? `, groups=${p.groups}` : ""}${p.bias ? "" : ", bias=False"})`,
  },
});

const depthwiseParams = z.object({
  kernel: pair,
  stride: pair.default([1, 1]),
  padding: padding.default("valid"),
  dilation: pair.default([1, 1]),
  depthMultiplier: posInt.default(1),
  bias: z.boolean().default(true),
  activation: activation.default("linear"),
});

type DepthwiseParams = z.infer<typeof depthwiseParams>;

export const depthwiseConv2d = defineOp<DepthwiseParams>({
  op: "depthwise_conv2d",
  family: "conv",
  title: "Depthwise Convolution",
  aliases: { keras: ["DepthwiseConv2D"], functional: ["tf.nn.depthwise_conv2d"] },
  params: depthwiseParams,
  defaults: { keras: { padding: "valid" }, pytorch: { padding: [0, 0] } },
  signatures: {
    keras: {
      positional: ["kernel_size", "strides", "padding", "depth_multiplier"],
      keys: {
        kernel_size: "kernel",
        strides: "stride",
        padding: "padding",
        depth_multiplier: "depthMultiplier",
        dilation_rate: "dilation",
        use_bias: "bias",
        activation: "activation",
      },
    },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([input], p, ctx) => {
    const { h, w, c } = image(input!);
    return imageLike(
      input!,
      windowOut(h, p.kernel[0], p.stride[0], padAt(p.padding, 0), p.dilation[0], ctx),
      windowOut(w, p.kernel[1], p.stride[1], padAt(p.padding, 1), p.dilation[1], ctx),
      c * p.depthMultiplier,
    );
  },
  countParams: ([input], p) => {
    const out = image(input!).c * p.depthMultiplier;
    return { trainable: p.kernel[0] * p.kernel[1] * out + (p.bias ? out : 0), nonTrainable: 0 };
  },
  flops: (_inputs, out, p) => {
    const o = image(out, "output");
    return 2 * o.h * o.w * p.kernel[0] * p.kernel[1] * o.c;
  },
  glyph: "volume",
  visualizer: "conv-sweep",
  explain: (p, io) =>
    `Each of the ${image(io.in[0]!).c} input channels gets its own ${times(p.kernel)} filter, so channels are filtered separately and not mixed.`,
  codegen: {
    keras: (p) =>
      `layers.DepthwiseConv2D(${kerasArgs([
        tuple(p.kernel),
        `strides=${tuple(p.stride)}`,
        typeof p.padding === "string" ? `padding="${p.padding}"` : null,
        p.depthMultiplier > 1 && `depth_multiplier=${p.depthMultiplier}`,
        !p.bias && "use_bias=False",
        kerasActivation(p.activation),
      ])})`,
    pytorch: (p, io) => {
      const { c } = image(io.in[0]!);
      return `nn.Conv2d(${c}, ${c * p.depthMultiplier}, kernel_size=${tuple(p.kernel)}, stride=${tuple(p.stride)}, padding=${pytorchPadding(p.padding)}, groups=${c}${p.bias ? "" : ", bias=False"})`;
    },
  },
});

export const convOps = [conv2d, depthwiseConv2d];
