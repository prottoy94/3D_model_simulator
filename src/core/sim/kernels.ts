/**
 * The arithmetic of each layer, in plain TypeScript on Float32Arrays.
 *
 * Weights are random, drawn the way Keras initializes an untrained layer
 * (Glorot uniform, zero bias) from a seed that depends only on the node's id,
 * so every run of the same model gives the same numbers. BatchNorm uses its
 * initial statistics (mean 0, variance 1). This is real computation on the
 * sample input, with untrained weights: the UI must always say so.
 */
import type { RuntimeNode } from "../ir/runtime";
import type { OpKind } from "../ops/kinds";
import type { Framework, TensorShape } from "../shapes/tensor";
import { EXTRA_KERNELS } from "./kernels-extra";
import {
  canonicalAxis,
  fillUniform,
  glorotLimit,
  hashString,
  randomStream,
  sizeOf,
  type SimTensor,
} from "./tensor";

/** Extra data some layers expose for their visualizers. */
export type KernelExtras = {
  /** Attention weights, one tokens×tokens matrix per head, row-major. */
  attention?: { heads: number; tokens: number; data: Float32Array };
  /** The first filter's weights for its first input channel (conv layers). */
  kernel?: { height: number; width: number; data: Float32Array };
};

export type KernelContext = {
  framework: Framework;
  /** The shape the engine computed for this node, channel-last, no batch. */
  expected: number[];
  /** Native shapes of the node's inputs, for axis conversion. */
  nativeInputs: TensorShape[];
  seed: number;
  extras: KernelExtras;
};

export type Kernel = (
  inputs: SimTensor[],
  params: Record<string, unknown>,
  ctx: KernelContext,
) => SimTensor;

export const tensor = (shape: number[], data: Float32Array): SimTensor => ({
  shape,
  data,
  illustrative: false,
});
const pairOf = (value: unknown, fallback: [number, number]): [number, number] =>
  Array.isArray(value) && value.length === 2 ? [Number(value[0]), Number(value[1])] : fallback;

// ── Activations ───────────────────────────────────────────────────────

function softmaxInPlace(data: Float32Array, size: number): void {
  for (let start = 0; start < data.length; start += size) {
    let max = -Infinity;
    for (let i = 0; i < size; i++) max = Math.max(max, data[start + i]!);
    let sum = 0;
    for (let i = 0; i < size; i++) {
      const value = Math.exp(data[start + i]! - max);
      data[start + i] = value;
      sum += value;
    }
    for (let i = 0; i < size; i++) data[start + i]! /= sum;
  }
}

/** Applies an activation by name. Softmax works over the last axis. */
export function activate(data: Float32Array, name: string, lastAxis: number): void {
  switch (name.toLowerCase()) {
    case "linear":
    case "":
      return;
    case "relu":
      for (let i = 0; i < data.length; i++) if (data[i]! < 0) data[i] = 0;
      return;
    case "gelu":
      for (let i = 0; i < data.length; i++) {
        const x = data[i]!;
        data[i] = 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
      }
      return;
    case "sigmoid":
      for (let i = 0; i < data.length; i++) data[i] = 1 / (1 + Math.exp(-data[i]!));
      return;
    case "tanh":
      for (let i = 0; i < data.length; i++) data[i] = Math.tanh(data[i]!);
      return;
    case "swish":
    case "silu":
      for (let i = 0; i < data.length; i++) data[i] = data[i]! / (1 + Math.exp(-data[i]!));
      return;
    case "hardswish":
      for (let i = 0; i < data.length; i++) {
        const x = data[i]!;
        data[i] = (x * Math.min(6, Math.max(0, x + 3))) / 6;
      }
      return;
    case "leaky_relu":
      for (let i = 0; i < data.length; i++) if (data[i]! < 0) data[i]! *= 0.2;
      return;
    case "softmax":
      softmaxInPlace(data, lastAxis);
      return;
    default:
      return; // unknown activations are left linear; the node keeps its flag from the engine
  }
}

const elementwise =
  (name: string): Kernel =>
  ([x]) => {
    const data = Float32Array.from(x!.data);
    activate(data, name, x!.shape[x!.shape.length - 1] ?? 1);
    return tensor(x!.shape, data);
  };

/**
 * Dot product of two runs of floats. Four running sums let the JavaScript
 * engine keep more of the work in registers; this loop is where almost all of
 * a CNN's simulation time goes.
 */
function dot(
  a: Float32Array,
  aStart: number,
  b: Float32Array,
  bStart: number,
  length: number,
): number {
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let i = 0;
  for (const end = length - 3; i < end; i += 4) {
    s0 += a[aStart + i]! * b[bStart + i]!;
    s1 += a[aStart + i + 1]! * b[bStart + i + 1]!;
    s2 += a[aStart + i + 2]! * b[bStart + i + 2]!;
    s3 += a[aStart + i + 3]! * b[bStart + i + 3]!;
  }
  for (; i < length; i++) s0 += a[aStart + i]! * b[bStart + i]!;
  return s0 + s1 + s2 + s3;
}

// ── Convolution and pooling ───────────────────────────────────────────

/** Rows of padding before the first input row, from the output size the engine computed. */
function padBefore(
  pad: unknown,
  axis: 0 | 1,
  input: number,
  output: number,
  kernel: number,
  stride: number,
  dilation: number,
): number {
  if (Array.isArray(pad)) return Number(pad[axis]);
  if (pad !== "same") return 0;
  const total = Math.max(0, (output - 1) * stride + (kernel - 1) * dilation + 1 - input);
  return Math.floor(total / 2);
}

const conv2d: Kernel = ([x], p, ctx) => {
  const [h, w, cin] = x!.shape as [number, number, number];
  const [oh, ow, cout] = ctx.expected as [number, number, number];
  const [kh, kw] = pairOf(p.kernel, [1, 1]);
  const [sh, sw] = pairOf(p.stride, [1, 1]);
  const [dh, dw] = pairOf(p.dilation, [1, 1]);
  const groups = Number(p.groups ?? 1);
  const cinGroup = cin / groups;
  const coutGroup = cout / groups;
  const pt = padBefore(p.padding, 0, h, oh, kh, sh, dh);
  const pl = padBefore(p.padding, 1, w, ow, kw, sw, dw);

  const patchSize = kh * kw * cinGroup;
  // Weights are laid out [filter][ky][kx][channel], so each filter is one contiguous row.
  const weights = fillUniform(
    new Float32Array(cout * patchSize),
    ctx.seed,
    glorotLimit(patchSize, kh * kw * coutGroup),
  );
  ctx.extras.kernel = { height: kh, width: kw, data: new Float32Array(kh * kw) };
  for (let i = 0; i < kh * kw; i++) ctx.extras.kernel.data[i] = weights[i * cinGroup]!;

  const out = new Float32Array(oh * ow * cout);
  const src = x!.data;
  // Positions are handled in blocks: gather a block of windows, then run each filter over
  // the whole block. A filter's weights then stay in the processor cache while they are used.
  const positions = oh * ow;
  const block = Math.max(8, Math.min(256, Math.floor(131_072 / patchSize)));
  const patches = new Float32Array(block * patchSize);
  for (let group = 0; group < groups; group++) {
    const channelStart = group * cinGroup;
    for (let first = 0; first < positions; first += block) {
      const count = Math.min(block, positions - first);
      let at = 0;
      for (let n = 0; n < count; n++) {
        const oy = Math.floor((first + n) / ow);
        const ox = (first + n) % ow;
        for (let ky = 0; ky < kh; ky++) {
          const iy = oy * sh - pt + ky * dh;
          for (let kx = 0; kx < kw; kx++) {
            const ix = ox * sw - pl + kx * dw;
            if (iy < 0 || iy >= h || ix < 0 || ix >= w) {
              for (let c = 0; c < cinGroup; c++) patches[at++] = 0;
            } else {
              const base = (iy * w + ix) * cin + channelStart;
              for (let c = 0; c < cinGroup; c++) patches[at++] = src[base + c]!;
            }
          }
        }
      }
      for (let f = 0; f < coutGroup; f++) {
        const row = (group * coutGroup + f) * patchSize;
        const channel = group * coutGroup + f;
        for (let n = 0; n < count; n++) {
          out[(first + n) * cout + channel] = dot(patches, n * patchSize, weights, row, patchSize);
        }
      }
    }
  }
  activate(out, String(p.activation ?? "linear"), cout);
  return tensor([oh, ow, cout], out);
};

const depthwiseConv2d: Kernel = ([x], p, ctx) => {
  const [h, w, cin] = x!.shape as [number, number, number];
  const [oh, ow, cout] = ctx.expected as [number, number, number];
  const [kh, kw] = pairOf(p.kernel, [1, 1]);
  const [sh, sw] = pairOf(p.stride, [1, 1]);
  const [dh, dw] = pairOf(p.dilation, [1, 1]);
  const multiplier = cout / cin;
  const pt = padBefore(p.padding, 0, h, oh, kh, sh, dh);
  const pl = padBefore(p.padding, 1, w, ow, kw, sw, dw);
  const weights = fillUniform(
    new Float32Array(kh * kw * cout),
    ctx.seed,
    glorotLimit(kh * kw * cin, kh * kw * cout),
  );
  const out = new Float32Array(oh * ow * cout);
  for (let oy = 0; oy < oh; oy++) {
    for (let ox = 0; ox < ow; ox++) {
      for (let ky = 0; ky < kh; ky++) {
        const iy = oy * sh - pt + ky * dh;
        if (iy < 0 || iy >= h) continue;
        for (let kx = 0; kx < kw; kx++) {
          const ix = ox * sw - pl + kx * dw;
          if (ix < 0 || ix >= w) continue;
          const inBase = (iy * w + ix) * cin;
          const outBase = (oy * ow + ox) * cout;
          const wBase = (ky * kw + kx) * cout;
          for (let c = 0; c < cout; c++) {
            out[outBase + c]! +=
              x!.data[inBase + Math.floor(c / multiplier)]! * weights[wBase + c]!;
          }
        }
      }
    }
  }
  activate(out, String(p.activation ?? "linear"), cout);
  return tensor([oh, ow, cout], out);
};

const pool =
  (kind: "max" | "avg"): Kernel =>
  ([x], p, ctx) => {
    const [h, w, c] = x!.shape as [number, number, number];
    const [oh, ow] = ctx.expected as [number, number, number];
    const [kh, kw] = pairOf(p.pool, [2, 2]);
    const [sh, sw] = pairOf(p.stride, [kh, kw]);
    const pt = padBefore(p.padding, 0, h, oh, kh, sh, 1);
    const pl = padBefore(p.padding, 1, w, ow, kw, sw, 1);
    const out = new Float32Array(oh * ow * c);
    for (let oy = 0; oy < oh; oy++) {
      for (let ox = 0; ox < ow; ox++) {
        const outBase = (oy * ow + ox) * c;
        for (let ch = 0; ch < c; ch++) {
          let best = kind === "max" ? -Infinity : 0;
          let count = 0;
          for (let ky = 0; ky < kh; ky++) {
            const iy = oy * sh - pt + ky;
            if (iy < 0 || iy >= h) continue;
            for (let kx = 0; kx < kw; kx++) {
              const ix = ox * sw - pl + kx;
              if (ix < 0 || ix >= w) continue;
              const value = x!.data[(iy * w + ix) * c + ch]!;
              if (kind === "max") best = Math.max(best, value);
              else best += value;
              count++;
            }
          }
          out[outBase + ch] = kind === "max" ? (count > 0 ? best : 0) : best / Math.max(1, count);
        }
      }
    }
    return tensor([oh, ow, c], out);
  };

const globalAvgPool: Kernel = ([x], _p, ctx) => {
  const c = x!.shape[x!.shape.length - 1]!;
  const positions = x!.data.length / c;
  const out = new Float32Array(c);
  for (let p = 0; p < positions; p++)
    for (let ch = 0; ch < c; ch++) out[ch]! += x!.data[p * c + ch]!;
  for (let ch = 0; ch < c; ch++) out[ch]! /= positions;
  return tensor(ctx.expected, out);
};

// ── Normalization ─────────────────────────────────────────────────────

const batchnorm: Kernel = ([x], p) => {
  // Untrained: mean 0, variance 1, γ 1, β 0. Values pass through, divided by sqrt(1 + ε).
  const scale = 1 / Math.sqrt(1 + Number(p.epsilon ?? 1e-3));
  const data = new Float32Array(x!.data.length);
  for (let i = 0; i < data.length; i++) data[i] = x!.data[i]! * scale;
  return tensor(x!.shape, data);
};

const layernorm: Kernel = ([x], p) => {
  const size = x!.shape[x!.shape.length - 1]!;
  const epsilon = Number(p.epsilon ?? 1e-3);
  const data = new Float32Array(x!.data.length);
  for (let start = 0; start < data.length; start += size) {
    let mean = 0;
    for (let i = 0; i < size; i++) mean += x!.data[start + i]!;
    mean /= size;
    let variance = 0;
    for (let i = 0; i < size; i++) variance += (x!.data[start + i]! - mean) ** 2;
    const scale = 1 / Math.sqrt(variance / size + epsilon);
    for (let i = 0; i < size; i++) data[start + i] = (x!.data[start + i]! - mean) * scale;
  }
  return tensor(x!.shape, data);
};

// ── Dense, embeddings, patches ────────────────────────────────────────

/**
 * y = x · W over the last axis. The weight matrix is never stored: each
 * output unit's row is generated from the seed, used for every position and
 * dropped, so even AlexNet's 26-million-weight layer needs no extra memory.
 */
export function linear(x: Float32Array, inSize: number, units: number, seed: number): Float32Array {
  const positions = x.length / inSize;
  const out = new Float32Array(positions * units);
  const row = new Float32Array(inSize);
  const limit = glorotLimit(inSize, units);
  for (let unit = 0; unit < units; unit++) {
    fillUniform(row, (seed + Math.imul(unit + 1, 0x9e3779b1)) >>> 0, limit);
    for (let p = 0; p < positions; p++) {
      const base = p * inSize;
      out[p * units + unit] = dot(x, base, row, 0, inSize);
    }
  }
  return out;
}

const dense: Kernel = ([x], p, ctx) => {
  const inSize = x!.shape[x!.shape.length - 1]!;
  const units = Number(p.units);
  const out = linear(x!.data, inSize, units, ctx.seed);
  activate(out, String(p.activation ?? "linear"), units);
  return tensor([...x!.shape.slice(0, -1), units], out);
};

function patches(x: SimTensor, ph: number, pw: number): SimTensor {
  const [h, w, c] = x.shape as [number, number, number];
  const rows = Math.floor(h / ph);
  const cols = Math.floor(w / pw);
  const size = ph * pw * c;
  const out = new Float32Array(rows * cols * size);
  let at = 0;
  for (let r = 0; r < rows; r++) {
    for (let col = 0; col < cols; col++) {
      for (let y = 0; y < ph; y++) {
        const base = ((r * ph + y) * w + col * pw) * c;
        for (let i = 0; i < pw * c; i++) out[at++] = x.data[base + i]!;
      }
    }
  }
  return tensor([rows * cols, size], out);
}

const patchify: Kernel = ([x], p) => patches(x!, ...pairOf(p.patch, [16, 16]));

const patchEmbed: Kernel = ([x], p, ctx) => {
  const tokens = x!.shape.length === 3 ? patches(x!, ...pairOf(p.patch, [16, 16])) : x!;
  const inSize = tokens.shape[1]!;
  const dim = Number(p.dim);
  return tensor([tokens.shape[0]!, dim], linear(tokens.data, inSize, dim, ctx.seed));
};

const posEmbed: Kernel = ([x], _p, ctx) => {
  // Keras Embedding starts uniform in ±0.05.
  const table = fillUniform(new Float32Array(x!.data.length), ctx.seed, 0.05);
  for (let i = 0; i < table.length; i++) table[i]! += x!.data[i]!;
  return tensor(x!.shape, table);
};

const embedding: Kernel = ([x], p, ctx) => {
  const dim = Number(p.dim);
  const out = new Float32Array(x!.data.length * dim);
  for (let i = 0; i < x!.data.length; i++) {
    const row = fillUniform(
      new Float32Array(dim),
      (ctx.seed + Math.imul(Math.round(x!.data[i]!) + 1, 0x9e3779b1)) >>> 0,
      0.05,
    );
    out.set(row, i * dim);
  }
  return tensor([...x!.shape, dim], out);
};

// ── Attention ─────────────────────────────────────────────────────────

const multiHeadAttention: Kernel = (inputs, p, ctx) => {
  const query = inputs[0]!;
  const value = inputs[1] ?? query;
  const key = inputs[2] ?? value;
  const tokens = query.shape[0]!;
  const sources = key.shape[0]!;
  const d = query.shape[1]!;
  const heads = Number(p.numHeads);
  const keyDim = ctx.framework === "pytorch" ? d / heads : Number(p.keyDim ?? d / heads);
  const valueDim = ctx.framework === "pytorch" ? keyDim : Number(p.valueDim ?? keyDim);
  const outDim = ctx.expected[ctx.expected.length - 1]!;

  const q = linear(query.data, d, heads * keyDim, ctx.seed ^ 0x51);
  const k = linear(key.data, key.shape[1]!, heads * keyDim, ctx.seed ^ 0x52);
  const v = linear(value.data, value.shape[1]!, heads * valueDim, ctx.seed ^ 0x53);

  const attention = new Float32Array(heads * tokens * sources);
  const mixed = new Float32Array(tokens * heads * valueDim);
  const scale = 1 / Math.sqrt(keyDim);
  for (let head = 0; head < heads; head++) {
    for (let t = 0; t < tokens; t++) {
      const rowStart = (head * tokens + t) * sources;
      const qBase = t * heads * keyDim + head * keyDim;
      for (let s = 0; s < sources; s++) {
        const kBase = s * heads * keyDim + head * keyDim;
        let score = 0;
        for (let i = 0; i < keyDim; i++) score += q[qBase + i]! * k[kBase + i]!;
        attention[rowStart + s] = score * scale;
      }
      softmaxInPlace(attention.subarray(rowStart, rowStart + sources), sources);
      const outBase = t * heads * valueDim + head * valueDim;
      for (let s = 0; s < sources; s++) {
        const weight = attention[rowStart + s]!;
        const vBase = s * heads * valueDim + head * valueDim;
        for (let i = 0; i < valueDim; i++) mixed[outBase + i]! += weight * v[vBase + i]!;
      }
    }
  }
  if (tokens === sources) ctx.extras.attention = { heads, tokens, data: attention };
  return tensor([tokens, outDim], linear(mixed, heads * valueDim, outDim, ctx.seed ^ 0x54));
};

// ── Reshape and merge ─────────────────────────────────────────────────

const reshapeTo: Kernel = ([x], _p, ctx) => tensor(ctx.expected, x!.data);

const slice: Kernel = ([x], p, ctx) => {
  const axis = canonicalAxis(Number(p.axis ?? 1), ctx.nativeInputs[0]!);
  const shape = x!.shape;
  const inner = sizeOf(shape.slice(axis + 1));
  const outer = sizeOf(shape.slice(0, axis));
  const size = shape[axis]!;
  const clamp = (value: number) => Math.max(0, Math.min(size, value < 0 ? size + value : value));
  const start = p.index !== undefined ? clamp(Number(p.index)) : clamp(Number(p.start ?? 0));
  const end = p.index !== undefined ? start + 1 : clamp(Number(p.end ?? size));
  const length = end - start;
  const out = new Float32Array(outer * length * inner);
  for (let o = 0; o < outer; o++) {
    out.set(
      x!.data.subarray((o * size + start) * inner, (o * size + end) * inner),
      o * length * inner,
    );
  }
  return tensor(ctx.expected, out);
};

/** NumPy-style broadcasting, aligned from the last axis. */
const broadcast =
  (combine: (a: number, b: number) => number): Kernel =>
  (inputs, _p, ctx) => {
    const shape = ctx.expected;
    const rank = shape.length;
    const out = new Float32Array(sizeOf(shape));
    const index = new Array<number>(rank).fill(0);
    const strides = inputs.map((input) => {
      const padded = [...new Array<number>(rank - input.shape.length).fill(1), ...input.shape];
      const result = new Array<number>(rank).fill(0);
      let step = 1;
      for (let axis = rank - 1; axis >= 0; axis--) {
        result[axis] = padded[axis] === 1 ? 0 : step;
        step *= padded[axis]!;
      }
      return result;
    });
    for (let flat = 0; flat < out.length; flat++) {
      let value = 0;
      for (let i = 0; i < inputs.length; i++) {
        let offset = 0;
        for (let axis = 0; axis < rank; axis++) offset += index[axis]! * strides[i]![axis]!;
        const item = inputs[i]!.data[offset]!;
        value = i === 0 ? item : combine(value, item);
      }
      out[flat] = value;
      for (let axis = rank - 1; axis >= 0 && ++index[axis]! === shape[axis]; axis--)
        index[axis] = 0;
    }
    return tensor(shape, out);
  };

const concat: Kernel = (inputs, p, ctx) => {
  const axis = canonicalAxis(Number(p.axis ?? -1), ctx.nativeInputs[0]!);
  const first = inputs[0]!;
  const outer = sizeOf(first.shape.slice(0, axis));
  const out = new Float32Array(sizeOf(ctx.expected));
  let at = 0;
  for (let o = 0; o < outer; o++) {
    for (const input of inputs) {
      const chunk = sizeOf(input.shape.slice(axis));
      out.set(input.data.subarray(o * chunk, (o + 1) * chunk), at);
      at += chunk;
    }
  }
  return tensor(ctx.expected, out);
};

// ── Detection ─────────────────────────────────────────────────────────

const roiPool: Kernel = ([features, boxes], _p, ctx) => {
  const [h, w, c] = features!.shape as [number, number, number];
  const [rois, ph, pw] = ctx.expected as [number, number, number, number];
  const out = new Float32Array(sizeOf(ctx.expected));
  for (let r = 0; r < rois; r++) {
    // Sample boxes are fractions of the image: [x1, y1, x2, y2].
    const [x1, y1, x2, y2] = [0, 1, 2, 3].map((i) =>
      Math.min(1, Math.max(0, boxes!.data[r * 4 + i] ?? 0)),
    );
    for (let py = 0; py < ph; py++) {
      const sy = Math.min(h - 1, Math.round((y1! + ((py + 0.5) / ph) * (y2! - y1!)) * h - 0.5));
      for (let px = 0; px < pw; px++) {
        const sx = Math.min(w - 1, Math.round((x1! + ((px + 0.5) / pw) * (x2! - x1!)) * w - 0.5));
        const from = (Math.max(0, sy) * w + Math.max(0, sx)) * c;
        out.set(features!.data.subarray(from, from + c), ((r * ph + py) * pw + px) * c);
      }
    }
  }
  return tensor(ctx.expected, out);
};

/** Ops the simulator can compute. Anything else is drawn with illustrative values. */
export const KERNELS: Partial<Record<OpKind, Kernel>> = {
  ...EXTRA_KERNELS,
  conv2d,
  depthwise_conv2d: depthwiseConv2d,
  maxpool2d: pool("max"),
  avgpool2d: pool("avg"),
  global_avg_pool: globalAvgPool,
  batchnorm,
  layernorm,
  relu: elementwise("relu"),
  gelu: elementwise("gelu"),
  sigmoid: elementwise("sigmoid"),
  softmax: elementwise("softmax"),
  dense,
  embedding,
  patchify,
  patch_embed: patchEmbed,
  pos_embed: posEmbed,
  multi_head_attention: multiHeadAttention,
  flatten: reshapeTo,
  reshape: reshapeTo,
  slice,
  add: broadcast((a, b) => a + b),
  multiply: broadcast((a, b) => a * b),
  concat,
  dropout: ([x]) => x!, // does nothing at inference
  roi_pool: roiPool,
};

/** Runs one node's kernel, once per item when it is wrapped in TimeDistributed. */
export function runKernel(
  node: RuntimeNode,
  inputs: SimTensor[],
  ctx: KernelContext,
): SimTensor | null {
  const kernel = KERNELS[node.op];
  if (!kernel) return null;
  if (node.wrapper !== "time_distributed") return kernel(inputs, node.params, ctx);

  const items = inputs[0]!.shape[0]!;
  const itemShape = ctx.expected.slice(1);
  const out = new Float32Array(sizeOf(ctx.expected));
  for (let item = 0; item < items; item++) {
    const sliced = inputs.map((input) => {
      const size = input.data.length / items;
      return tensor(input.shape.slice(1), input.data.subarray(item * size, (item + 1) * size));
    });
    const result = kernel(sliced, node.params, { ...ctx, expected: itemShape });
    out.set(result.data, item * sizeOf(itemShape));
  }
  return tensor(ctx.expected, out);
}

/** A deterministic seed for a node's weights. Unrolled copies of a block get different weights. */
export const seedFor = (nodeId: string) => hashString(nodeId);

export { randomStream };
