/**
 * Compute for the Phase 8 layers. Same rules as kernels.ts: real arithmetic on
 * the sample, untrained seeded weights, channel-last tensors without a batch.
 */
import type { OpKind } from "../ops/kinds";
import { suppress } from "./boxes";
import { activate, linear, tensor, type Kernel } from "./kernels";
import { canonicalAxis, fillUniform, glorotLimit, sizeOf } from "./tensor";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const elementwise =
  (name: string): Kernel =>
  ([x]) => {
    const data = Float32Array.from(x!.data);
    activate(data, name, x!.shape[x!.shape.length - 1] ?? 1);
    return tensor(x!.shape, data);
  };

const groupnorm: Kernel = ([x], p) => {
  const channels = x!.shape[x!.shape.length - 1]!;
  const groups = Number(p.groups ?? 32);
  const size = channels / groups;
  const positions = x!.data.length / channels;
  const epsilon = Number(p.epsilon ?? 1e-5);
  const out = new Float32Array(x!.data.length);
  for (let g = 0; g < groups; g++) {
    let mean = 0;
    for (let pos = 0; pos < positions; pos++) {
      for (let c = 0; c < size; c++) mean += x!.data[pos * channels + g * size + c]!;
    }
    mean /= positions * size;
    let variance = 0;
    for (let pos = 0; pos < positions; pos++) {
      for (let c = 0; c < size; c++)
        variance += (x!.data[pos * channels + g * size + c]! - mean) ** 2;
    }
    const scale = 1 / Math.sqrt(variance / (positions * size) + epsilon);
    for (let pos = 0; pos < positions; pos++) {
      for (let c = 0; c < size; c++) {
        const at = pos * channels + g * size + c;
        out[at] = (x!.data[at]! - mean) * scale;
      }
    }
  }
  return tensor(x!.shape, out);
};

/** Per-channel average and maximum over every position. */
function channelSummaries(
  data: Float32Array,
  channels: number,
): { avg: Float32Array; max: Float32Array } {
  const positions = data.length / channels;
  const avg = new Float32Array(channels);
  const max = new Float32Array(channels).fill(-Infinity);
  for (let pos = 0; pos < positions; pos++) {
    for (let c = 0; c < channels; c++) {
      const value = data[pos * channels + c]!;
      avg[c]! += value;
      if (value > max[c]!) max[c] = value;
    }
  }
  for (let c = 0; c < channels; c++) avg[c]! /= positions;
  return { avg, max };
}

const globalMaxPool: Kernel = ([x], _p, ctx) =>
  tensor(ctx.expected, channelSummaries(x!.data, x!.shape[x!.shape.length - 1]!).max);

const adaptiveAvgPool2d: Kernel = ([x], _p, ctx) => {
  const [h, w, c] = x!.shape as [number, number, number];
  const [oh, ow] = ctx.expected as [number, number, number];
  const out = new Float32Array(oh * ow * c);
  for (let oy = 0; oy < oh; oy++) {
    const y0 = Math.floor((oy * h) / oh);
    const y1 = Math.ceil(((oy + 1) * h) / oh);
    for (let ox = 0; ox < ow; ox++) {
      const x0 = Math.floor((ox * w) / ow);
      const x1 = Math.ceil(((ox + 1) * w) / ow);
      const count = (y1 - y0) * (x1 - x0);
      for (let iy = y0; iy < y1; iy++) {
        for (let ix = x0; ix < x1; ix++) {
          for (let ch = 0; ch < c; ch++)
            out[(oy * ow + ox) * c + ch]! += x!.data[(iy * w + ix) * c + ch]! / count;
        }
      }
    }
  }
  return tensor(ctx.expected, out);
};

const upsample: Kernel = ([x], p, ctx) => {
  const [h, w, c] = x!.shape as [number, number, number];
  const [oh, ow] = ctx.expected as [number, number, number];
  const out = new Float32Array(oh * ow * c);
  const bilinear = p.mode === "bilinear";
  for (let oy = 0; oy < oh; oy++) {
    // Pixel centres are aligned (align_corners=False), as in PyTorch and TensorFlow.
    const fy = Math.min(h - 1, Math.max(0, ((oy + 0.5) * h) / oh - 0.5));
    const y0 = bilinear ? Math.floor(fy) : Math.min(h - 1, Math.floor((oy * h) / oh));
    const y1 = Math.min(h - 1, y0 + 1);
    const wy = bilinear ? fy - y0 : 0;
    for (let ox = 0; ox < ow; ox++) {
      const fx = Math.min(w - 1, Math.max(0, ((ox + 0.5) * w) / ow - 0.5));
      const x0 = bilinear ? Math.floor(fx) : Math.min(w - 1, Math.floor((ox * w) / ow));
      const x1 = Math.min(w - 1, x0 + 1);
      const wx = bilinear ? fx - x0 : 0;
      const at = (oy * ow + ox) * c;
      for (let ch = 0; ch < c; ch++) {
        const top =
          x!.data[(y0 * w + x0) * c + ch]! * (1 - wx) + x!.data[(y0 * w + x1) * c + ch]! * wx;
        const bottom =
          x!.data[(y1 * w + x0) * c + ch]! * (1 - wx) + x!.data[(y1 * w + x1) * c + ch]! * wx;
        out[at + ch] = top * (1 - wy) + bottom * wy;
      }
    }
  }
  return tensor(ctx.expected, out);
};

const reduce =
  (kind: "mean" | "max"): Kernel =>
  ([x], p, ctx) => {
    const shape = x!.shape;
    const axes = new Set(
      (p.axis as number[]).map((axis) => canonicalAxis(axis, ctx.nativeInputs[0]!)),
    );
    const out = new Float32Array(sizeOf(ctx.expected)).fill(kind === "max" ? -Infinity : 0);
    const index = new Array<number>(shape.length).fill(0);
    let reduced = 1;
    for (const axis of axes) reduced *= shape[axis]!;
    for (let flat = 0; flat < x!.data.length; flat++) {
      // The output position is the input position with the reduced axes dropped.
      let target = 0;
      for (let axis = 0; axis < shape.length; axis++) {
        if (!axes.has(axis)) target = target * shape[axis]! + index[axis]!;
      }
      const value = x!.data[flat]!;
      if (kind === "max") {
        if (value > out[target]!) out[target] = value;
      } else out[target]! += value / reduced;
      for (let axis = shape.length - 1; axis >= 0 && ++index[axis]! === shape[axis]; axis--)
        index[axis] = 0;
    }
    return tensor(ctx.expected, out);
  };

/** Squeeze-and-excite, and CBAM's channel attention (which also uses the per-channel maximum). */
const channelGate =
  (cbam: boolean): Kernel =>
  ([x], p, ctx) => {
    const channels = x!.shape[x!.shape.length - 1]!;
    const hidden = Math.max(
      Math.floor(channels / Number(p.reduction ?? 16)),
      Number(p.minHidden ?? 1),
    );
    const { avg, max } = channelSummaries(x!.data, channels);
    // One small network, applied to both summaries with the same weights.
    const mlp = (summary: Float32Array) => {
      const inner = linear(summary, channels, hidden, ctx.seed ^ 0x61);
      activate(inner, "relu", hidden);
      return linear(inner, hidden, channels, ctx.seed ^ 0x62);
    };
    const scores = mlp(avg);
    if (cbam) {
      const fromMax = mlp(max);
      for (let c = 0; c < channels; c++) scores[c]! += fromMax[c]!;
    }
    const out = new Float32Array(x!.data.length);
    for (let i = 0; i < out.length; i++) out[i] = x!.data[i]! * sigmoid(scores[i % channels]!);
    return tensor(x!.shape, out);
  };

const spatialAttention: Kernel = ([x], p, ctx) => {
  const [h, w, c] = x!.shape as [number, number, number];
  const k = Number(p.kernel ?? 7);
  const pad = Math.floor(k / 2);
  // Two maps: the average and the maximum over the channels at each position.
  const summary = new Float32Array(h * w * 2);
  for (let pos = 0; pos < h * w; pos++) {
    let sum = 0;
    let max = -Infinity;
    for (let ch = 0; ch < c; ch++) {
      const value = x!.data[pos * c + ch]!;
      sum += value;
      if (value > max) max = value;
    }
    summary[pos * 2] = sum / c;
    summary[pos * 2 + 1] = max;
  }
  const weights = fillUniform(new Float32Array(k * k * 2), ctx.seed, glorotLimit(k * k * 2, k * k));
  const out = new Float32Array(x!.data.length);
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < w; xx++) {
      let score = 0;
      for (let ky = 0; ky < k; ky++) {
        const iy = y - pad + ky;
        if (iy < 0 || iy >= h) continue;
        for (let kx = 0; kx < k; kx++) {
          const ix = xx - pad + kx;
          if (ix < 0 || ix >= w) continue;
          const from = (iy * w + ix) * 2;
          const at = (ky * k + kx) * 2;
          score += summary[from]! * weights[at]! + summary[from + 1]! * weights[at + 1]!;
        }
      }
      const gate = sigmoid(score);
      const base = (y * w + xx) * c;
      for (let ch = 0; ch < c; ch++) out[base + ch] = x!.data[base + ch]! * gate;
    }
  }
  return tensor(x!.shape, out);
};

const GATES = { rnn: 1, lstm: 4, gru: 3 } as const;

/** One direction of a recurrent layer. Returns the memory after every step. */
function recur(
  kind: keyof typeof GATES,
  input: Float32Array,
  steps: number,
  size: number,
  units: number,
  seed: number,
  reverse: boolean,
): Float32Array {
  const gates = GATES[kind];
  const wx = fillUniform(
    new Float32Array(gates * units * size),
    seed ^ 0x71,
    glorotLimit(size, units),
  );
  const wh = fillUniform(
    new Float32Array(gates * units * units),
    seed ^ 0x72,
    glorotLimit(units, units),
  );
  const outputs = new Float32Array(steps * units);
  const h = new Float32Array(units);
  const cell = new Float32Array(units);
  const fromInput = new Float32Array(gates * units);
  const fromMemory = new Float32Array(gates * units);
  for (let n = 0; n < steps; n++) {
    const t = reverse ? steps - 1 - n : n;
    for (let row = 0; row < gates * units; row++) {
      let a = 0;
      for (let i = 0; i < size; i++) a += wx[row * size + i]! * input[t * size + i]!;
      let b = 0;
      for (let i = 0; i < units; i++) b += wh[row * units + i]! * h[i]!;
      fromInput[row] = a;
      fromMemory[row] = b;
    }
    for (let u = 0; u < units; u++) {
      if (kind === "rnn") {
        h[u] = Math.tanh(fromInput[u]! + fromMemory[u]!);
      } else if (kind === "lstm") {
        const gate = (g: number) => fromInput[g * units + u]! + fromMemory[g * units + u]!;
        // Gates in order: input, forget, candidate, output.
        cell[u] = sigmoid(gate(1)) * cell[u]! + sigmoid(gate(0)) * Math.tanh(gate(2));
        outputs[t * units + u] = sigmoid(gate(3)) * Math.tanh(cell[u]!);
      } else {
        const update = sigmoid(fromInput[u]! + fromMemory[u]!);
        const reset = sigmoid(fromInput[units + u]! + fromMemory[units + u]!);
        const candidate = Math.tanh(fromInput[2 * units + u]! + reset * fromMemory[2 * units + u]!);
        outputs[t * units + u] = (1 - update) * candidate + update * h[u]!;
      }
    }
    // The new memory is read only after every unit has used the old one.
    if (kind === "rnn") outputs.set(h, t * units);
    else h.set(outputs.subarray(t * units, (t + 1) * units));
  }
  return outputs;
}

const recurrent =
  (kind: keyof typeof GATES): Kernel =>
  ([x], p, ctx) => {
    const [steps, size] = x!.shape as [number, number];
    const units = Number(p.units);
    const directions = p.bidirectional ? 2 : 1;
    const forward = recur(kind, x!.data, steps, size, units, ctx.seed, false);
    const backward = p.bidirectional
      ? recur(kind, x!.data, steps, size, units, ctx.seed ^ 0x7f, true)
      : null;
    if (p.returnSequences) {
      const out = new Float32Array(steps * units * directions);
      for (let t = 0; t < steps; t++) {
        out.set(forward.subarray(t * units, (t + 1) * units), t * units * directions);
        if (backward)
          out.set(backward.subarray(t * units, (t + 1) * units), t * units * directions + units);
      }
      return tensor(ctx.expected, out);
    }
    // The final memory: the last step going forward, the first step going backward.
    const out = new Float32Array(units * directions);
    out.set(forward.subarray((steps - 1) * units, steps * units), 0);
    if (backward) out.set(backward.subarray(0, units), units);
    return tensor(ctx.expected, out);
  };

const anchorGenerator: Kernel = ([x], p, ctx) => {
  const [h, w] = x!.shape as [number, number, number];
  const perLocation = Number(p.perLocation ?? 9);
  const out = new Float32Array(sizeOf(ctx.expected));
  // Three aspect ratios, and as many scales as needed. Boxes are fractions of the image: x1, y1, x2, y2.
  const ratios = [0.5, 1, 2];
  let at = 0;
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < w; xx++) {
      const cx = (xx + 0.5) / w;
      const cy = (y + 0.5) / h;
      for (let a = 0; a < perLocation; a++) {
        const ratio = ratios[a % ratios.length]!;
        const scale = (4 / Math.max(h, w)) * 2 ** (Math.floor(a / ratios.length) / 3);
        const bw = (scale * Math.sqrt(ratio)) / 2;
        const bh = scale / Math.sqrt(ratio) / 2;
        out.set([cx - bw, cy - bh, cx + bw, cy + bh], at);
        at += 4;
      }
    }
  }
  return tensor(ctx.expected, out);
};

// ── Sequences ─────────────────────────────────────────────────────────

/**
 * A 1D layer's input as [length][channels]. PyTorch reads (batch, channels, length),
 * so a tensor that was stored the other way round is turned first.
 */
function sequenceInput(
  x: { shape: number[]; data: Float32Array },
  ctx: Parameters<Kernel>[2],
): { length: number; channels: number; data: Float32Array } {
  const [a, b] = x.shape as [number, number];
  const turned = ctx.framework === "pytorch" && ctx.nativeInputs[0]?.layout !== "NCL";
  if (!turned) return { length: a, channels: b, data: x.data };
  const data = new Float32Array(x.data.length);
  for (let c = 0; c < a; c++) for (let t = 0; t < b; t++) data[t * a + c] = x.data[c * b + t]!;
  return { length: b, channels: a, data };
}

function padBefore1d(
  pad: unknown,
  input: number,
  output: number,
  kernel: number,
  stride: number,
  dilation: number,
): number {
  if (typeof pad === "number") return pad;
  if (pad !== "same") return 0;
  return Math.floor(Math.max(0, (output - 1) * stride + (kernel - 1) * dilation + 1 - input) / 2);
}

const conv1d: Kernel = ([x], p, ctx) => {
  const { length, channels, data } = sequenceInput(x!, ctx);
  const [outLength, filters] = ctx.expected as [number, number];
  const kernel = Number(p.kernel ?? 1);
  const stride = Number(p.stride ?? 1);
  const dilation = Number(p.dilation ?? 1);
  const groups = Number(p.groups ?? 1);
  const inGroup = channels / groups;
  const outGroup = filters / groups;
  const before = padBefore1d(p.padding, length, outLength, kernel, stride, dilation);
  // Weights are laid out [filter][step][channel].
  const weights = fillUniform(
    new Float32Array(filters * kernel * inGroup),
    ctx.seed,
    glorotLimit(kernel * inGroup, kernel * outGroup),
  );
  const out = new Float32Array(outLength * filters);
  for (let t = 0; t < outLength; t++) {
    for (let f = 0; f < filters; f++) {
      const first = Math.floor(f / outGroup) * inGroup;
      let sum = 0;
      for (let k = 0; k < kernel; k++) {
        const at = t * stride - before + k * dilation;
        if (at < 0 || at >= length) continue;
        const row = (f * kernel + k) * inGroup;
        for (let c = 0; c < inGroup; c++) {
          sum += data[at * channels + first + c]! * weights[row + c]!;
        }
      }
      out[t * filters + f] = sum;
    }
  }
  activate(out, String(p.activation ?? "linear"), filters);
  return tensor(ctx.expected, out);
};

const maxpool1d: Kernel = ([x], p, ctx) => {
  const { length, channels, data } = sequenceInput(x!, ctx);
  const [outLength] = ctx.expected as [number, number];
  const pool = Number(p.pool ?? 2);
  const stride = Number(p.stride ?? pool);
  const before = padBefore1d(p.padding, length, outLength, pool, stride, 1);
  const out = new Float32Array(outLength * channels).fill(-Infinity);
  for (let t = 0; t < outLength; t++) {
    for (let k = 0; k < pool; k++) {
      const at = t * stride - before + k;
      if (at < 0 || at >= length) continue;
      for (let c = 0; c < channels; c++) {
        const value = data[at * channels + c]!;
        if (value > out[t * channels + c]!) out[t * channels + c] = value;
      }
    }
  }
  return tensor(ctx.expected, out);
};

// ── Detection after the heads ─────────────────────────────────────────

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

type Four = [number, number, number, number];
const four = (data: Float32Array, at: number): Four => [
  data[at] ?? 0,
  data[at + 1] ?? 0,
  data[at + 2] ?? 0,
  data[at + 3] ?? 0,
];

const roiAlign: Kernel = ([features, boxes], p, ctx) => {
  const [h, w, c] = features!.shape as [number, number, number];
  const [rois, ph, pw] = ctx.expected as [number, number, number, number];
  const samples = Number(p.samplingRatio ?? 2);
  const out = new Float32Array(sizeOf(ctx.expected));
  const src = features!.data;
  for (let r = 0; r < rois; r++) {
    // Sample boxes are fractions of the image: [x1, y1, x2, y2].
    const [x1, y1, x2, y2] = four(boxes!.data, r * 4).map(clamp01) as Four;
    for (let py = 0; py < ph; py++) {
      for (let px = 0; px < pw; px++) {
        const to = ((r * ph + py) * pw + px) * c;
        for (let sy = 0; sy < samples; sy++) {
          // Positions are continuous: no rounding to the grid, which is the point of RoI align.
          const fy = (y1 + ((py + (sy + 0.5) / samples) / ph) * (y2 - y1)) * h - 0.5;
          const y0 = Math.min(h - 1, Math.max(0, Math.floor(fy)));
          const yb = Math.min(h - 1, y0 + 1);
          const wy = clamp01(fy - y0);
          for (let sx = 0; sx < samples; sx++) {
            const fx = (x1 + ((px + (sx + 0.5) / samples) / pw) * (x2 - x1)) * w - 0.5;
            const x0 = Math.min(w - 1, Math.max(0, Math.floor(fx)));
            const xb = Math.min(w - 1, x0 + 1);
            const wx = clamp01(fx - x0);
            for (let ch = 0; ch < c; ch++) {
              out[to + ch]! +=
                src[(y0 * w + x0) * c + ch]! * (1 - wy) * (1 - wx) +
                src[(y0 * w + xb) * c + ch]! * (1 - wy) * wx +
                src[(yb * w + x0) * c + ch]! * wy * (1 - wx) +
                src[(yb * w + xb) * c + ch]! * wy * wx;
            }
          }
        }
        for (let ch = 0; ch < c; ch++) out[to + ch]! /= samples * samples;
      }
    }
  }
  return tensor(ctx.expected, out);
};

/** Largest stretch a predicted offset may ask for, as detection libraries limit it. */
const MAX_STRETCH = Math.log(1000 / 16);

const boxDecode: Kernel = ([deltas, anchors], p, ctx) => {
  const count = ctx.expected[0]!;
  const available = Math.max(1, Math.floor(anchors!.data.length / 4));
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const [ax1, ay1, ax2, ay2] = four(anchors!.data, (i % available) * 4);
    const width = ax2 - ax1;
    const height = ay2 - ay1;
    const [dx, dy, dw, dh] = four(deltas!.data, i * 4);
    const cx = ax1 + width / 2 + dx * width;
    const cy = ay1 + height / 2 + dy * height;
    const bw = width * Math.exp(Math.min(dw, MAX_STRETCH));
    const bh = height * Math.exp(Math.min(dh, MAX_STRETCH));
    const box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2];
    out.set(p.clip === false ? box : box.map(clamp01), i * 4);
  }
  return tensor(ctx.expected, out);
};

const nms: Kernel = ([boxes, scores], p, ctx) => {
  const count = Math.floor(boxes!.data.length / 4);
  const perBox = Math.max(1, Math.floor(scores!.data.length / count));
  // With one score per class, a box is as good as its best class.
  const best = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    let top = -Infinity;
    for (let k = 0; k < perBox; k++) top = Math.max(top, scores!.data[i * perBox + k] ?? -Infinity);
    best[i] = top;
  }
  // Untrained scores can all sit under the score threshold, so the sample ignores it
  // and still shows the strongest boxes that do not overlap.
  const kept = suppress(
    boxes!.data,
    best,
    Number(p.iouThreshold ?? 0.5),
    -Infinity,
    Math.min(Number(p.maxOutput ?? 100), ctx.expected[0]!),
  );
  const out = new Float32Array(sizeOf(ctx.expected));
  kept.forEach((index, row) => out.set(boxes!.data.subarray(index * 4, index * 4 + 4), row * 4));
  return tensor(ctx.expected, out);
};

export const EXTRA_KERNELS: Partial<Record<OpKind, Kernel>> = {
  conv1d,
  maxpool1d,
  roi_align: roiAlign,
  box_decode: boxDecode,
  nms,
  groupnorm,
  tanh: elementwise("tanh"),
  silu: elementwise("silu"),
  leaky_relu: elementwise("leaky_relu"),
  hardswish: elementwise("hardswish"),
  global_max_pool: globalMaxPool,
  adaptive_avg_pool2d: adaptiveAvgPool2d,
  upsample,
  reduce_mean: reduce("mean"),
  reduce_max: reduce("max"),
  squeeze_excite: channelGate(false),
  channel_attention: channelGate(true),
  spatial_attention: spatialAttention,
  rnn: recurrent("rnn"),
  lstm: recurrent("lstm"),
  gru: recurrent("gru"),
  anchor_generator: anchorGenerator,
};
