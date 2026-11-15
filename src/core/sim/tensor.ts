/**
 * Tensors as the simulator computes them: one sample (no batch axis), always
 * stored channel-last, whatever the framework's native layout is. A PyTorch
 * (B, C, H, W) tensor is held here as [H, W, C].
 */
import { isKnown, type TensorShape } from "../shapes/tensor";

export type SimTensor = {
  /** Sizes without the batch, channel-last. */
  shape: number[];
  data: Float32Array;
  /**
   * True when the values are not the result of real arithmetic on the input
   * (a declared layer, or anything computed from one). Shown as "illustrative".
   */
  illustrative: boolean;
};

/** Size used for a dimension only known at run time (e.g. the number of RoIs). */
export const RUNTIME_DIM = 4;

export const sizeOf = (shape: readonly number[]) => shape.reduce((total, dim) => total * dim, 1);

/** The engine's shape for a node, as the simulator stores it. */
export function canonicalShape(shape: TensorShape): number[] {
  const dims = shape.dims.slice(1).map((dim) => (isKnown(dim) ? dim : RUNTIME_DIM));
  if (shape.layout === "NCHW" && dims.length === 3) return [dims[1]!, dims[2]!, dims[0]!];
  if (shape.layout === "NCL" && dims.length === 2) return [dims[1]!, dims[0]!];
  return dims;
}

/**
 * Converts an axis written for the framework's layout (batch included, may be
 * negative) to an index into the simulator's channel-last shape.
 */
export function canonicalAxis(axis: number, native: TensorShape): number {
  const rank = native.dims.length;
  const index = axis < 0 ? rank + axis : axis;
  if (native.layout === "NCHW" && rank === 4) return [0, 2, 0, 1][index] ?? 0;
  if (native.layout === "NCL" && rank === 3) return [0, 1, 0][index] ?? 0;
  return Math.max(0, index - 1);
}

// ── Deterministic pseudo-random numbers ───────────────────────────────

/** A stable 32-bit hash of a string (FNV-1a), used to seed each layer's weights. */
export function hashString(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/** mulberry32: small, fast, and the same on every machine. Returns values in [0, 1). */
export function randomStream(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fills `out` with uniform values in [-limit, limit]. */
export function fillUniform(out: Float32Array, seed: number, limit: number): Float32Array {
  const next = randomStream(seed);
  for (let i = 0; i < out.length; i++) out[i] = (next() * 2 - 1) * limit;
  return out;
}

/** Glorot uniform, the default initializer of Keras Dense and Conv layers. */
export const glorotLimit = (fanIn: number, fanOut: number) => Math.sqrt(6 / (fanIn + fanOut));

/**
 * Stand-in values for a layer the simulator cannot compute: smooth, seeded
 * noise in the right shape. Always marked illustrative.
 */
export function illustrativeTensor(shape: number[], seed: number): SimTensor {
  const data = new Float32Array(sizeOf(shape));
  const next = randomStream(seed);
  const channels = shape[shape.length - 1] ?? 1;
  const phases = Array.from({ length: Math.min(channels, 64) }, () => [
    next() * 6.28,
    0.5 + next() * 3,
    next() * 6.28,
  ]);
  const positions = data.length / channels;
  const width = shape.length >= 3 ? shape[shape.length - 2]! : positions;
  for (let p = 0; p < positions; p++) {
    const x = (p % width) / Math.max(1, width);
    const y = Math.floor(p / width) / Math.max(1, positions / width);
    for (let c = 0; c < channels; c++) {
      const [a, f, b] = phases[c % phases.length]!;
      data[p * channels + c] = Math.max(
        0,
        Math.sin(a! + f! * 6.28 * x) * Math.cos(b! + f! * 4 * y) + (next() - 0.5) * 0.3,
      );
    }
  }
  return { shape, data, illustrative: true };
}
