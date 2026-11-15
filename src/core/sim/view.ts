/**
 * Turns a computed tensor into something small enough to show: statistics, a
 * histogram and a few pictures. The full tensor never leaves the worker.
 */
import type { NodeResult } from "./run";
import type { SimTensor } from "./tensor";

export type Picture = { width: number; height: number; rgba: Uint8ClampedArray };

export type TensorStats = {
  min: number;
  max: number;
  mean: number;
  std: number;
  zeroFraction: number;
};

export type TensorView = {
  nodeId: string;
  /** Sizes without the batch, channel-last. */
  shape: number[];
  kind: "image" | "tokens" | "vector";
  /** True when the values are not real arithmetic on the input. */
  illustrative: boolean;
  note: string | null;
  ms: number;
  stats: TensorStats;
  /** Counts in equal bins from stats.min to stats.max. */
  histogram: number[];
  /** A mosaic of channel maps (image), a token × feature heatmap (tokens), or null (vector). */
  picture: Picture | null;
  /** How many channel tiles the mosaic shows, out of how many, and its grid. */
  tiles: { shown: number; total: number; columns: number } | null;
  /** The values themselves for vectors, sampled evenly when there are more than 512. */
  values: Float32Array | null;
  /** True when the values are probabilities (non-negative, summing to 1). */
  probabilities: boolean;
  attention: { heads: number; tokens: number; pictures: Picture[] } | null;
  kernel: { height: number; width: number; data: Float32Array } | null;
  /** First channel of the first input and of the output, for the sliding-window visualizer. */
  focus: { input: Picture | null; output: Picture | null };
};

const HISTOGRAM_BINS = 24;
const MAX_TILES = 64;
const MAX_TILE = 32;
const MAX_VALUES = 512;

/** Black → blue → cyan → white. Zero is black, so mostly-empty maps stay dark and structure stands out. */
function heat(t: number, out: Uint8ClampedArray, at: number): void {
  const v = Math.sqrt(Math.min(1, Math.max(0, t))); // lifts faint values so they are visible
  out[at] = Math.round(255 * Math.max(0, v * 1.7 - 0.7));
  out[at + 1] = Math.round(255 * Math.min(1, v * 1.15));
  out[at + 2] = Math.round(255 * Math.min(1, v * 1.6));
  out[at + 3] = 255;
}

/**
 * The value below which 99% of the tensor lies, read off the histogram. Using
 * it as the top of the colour scale keeps one extreme value from turning the
 * whole picture dark.
 */
function displayMax(histogram: number[], stats: TensorStats): number {
  const total = histogram.reduce((a, b) => a + b, 0);
  let seen = 0;
  for (let bin = 0; bin < histogram.length; bin++) {
    seen += histogram[bin]!;
    if (seen >= total * 0.99)
      return stats.min + ((bin + 1) / histogram.length) * (stats.max - stats.min);
  }
  return stats.max;
}

function statsOf(data: Float32Array): TensorStats {
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let zeros = 0;
  for (let i = 0; i < data.length; i++) {
    const value = data[i]!;
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
    if (value === 0) zeros++;
  }
  const mean = sum / Math.max(1, data.length);
  let variance = 0;
  for (let i = 0; i < data.length; i++) variance += (data[i]! - mean) ** 2;
  if (!Number.isFinite(min)) min = max = 0;
  return {
    min,
    max,
    mean,
    std: Math.sqrt(variance / Math.max(1, data.length)),
    zeroFraction: zeros / Math.max(1, data.length),
  };
}

function histogramOf(data: Float32Array, stats: TensorStats): number[] {
  const bins = new Array<number>(HISTOGRAM_BINS).fill(0);
  const span = stats.max - stats.min || 1;
  // Large tensors are sampled; the shape of the distribution is what matters.
  const step = Math.max(1, Math.floor(data.length / 200_000));
  for (let i = 0; i < data.length; i += step) {
    bins[
      Math.min(HISTOGRAM_BINS - 1, Math.floor(((data[i]! - stats.min) / span) * HISTOGRAM_BINS))
    ]!++;
  }
  return bins;
}

/** One channel of an [H, W, C] tensor as a picture of at most `limit` pixels a side. */
function channelPicture(
  tensor: SimTensor,
  channel: number,
  limit: number,
  lo: number,
  hi: number,
): Picture {
  const [h, w, c] = tensor.shape as [number, number, number];
  const height = Math.min(h, limit);
  const width = Math.min(w, limit);
  const rgba = new Uint8ClampedArray(width * height * 4);
  const span = hi - lo || 1;
  for (let y = 0; y < height; y++) {
    const sy = Math.min(h - 1, Math.floor(((y + 0.5) / height) * h));
    for (let x = 0; x < width; x++) {
      const sx = Math.min(w - 1, Math.floor(((x + 0.5) / width) * w));
      heat((tensor.data[(sy * w + sx) * c + channel]! - lo) / span, rgba, (y * width + x) * 4);
    }
  }
  return { width, height, rgba };
}

function colorPicture(tensor: SimTensor, limit: number): Picture {
  const [h, w, c] = tensor.shape as [number, number, number];
  const height = Math.min(h, limit);
  const width = Math.min(w, limit);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const sy = Math.min(h - 1, Math.floor(((y + 0.5) / height) * h));
    for (let x = 0; x < width; x++) {
      const from = (sy * w + Math.min(w - 1, Math.floor(((x + 0.5) / width) * w))) * c;
      const at = (y * width + x) * 4;
      for (let i = 0; i < 3; i++)
        rgba[at + i] = Math.round(
          255 * Math.min(1, Math.max(0, tensor.data[from + (c === 3 ? i : 0)]!)),
        );
      rgba[at + 3] = 255;
    }
  }
  return { width, height, rgba };
}

/** Channel maps side by side in a grid, with a 1-pixel gap between tiles. */
function mosaic(
  tensor: SimTensor,
  stats: TensorStats,
  top: number,
): { picture: Picture; tiles: NonNullable<TensorView["tiles"]> } {
  const channels = tensor.shape[2]!;
  const shown = Math.min(channels, MAX_TILES);
  const columns = Math.ceil(Math.sqrt(shown));
  const rows = Math.ceil(shown / columns);
  const first = channelPicture(tensor, 0, MAX_TILE, stats.min, top);
  const tw = first.width + 1;
  const th = first.height + 1;
  const width = columns * tw - 1;
  const height = rows * th - 1;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let tile = 0; tile < shown; tile++) {
    const picture = tile === 0 ? first : channelPicture(tensor, tile, MAX_TILE, stats.min, top);
    const ox = (tile % columns) * tw;
    const oy = Math.floor(tile / columns) * th;
    for (let y = 0; y < picture.height; y++) {
      rgba.set(
        picture.rgba.subarray(y * picture.width * 4, (y + 1) * picture.width * 4),
        ((oy + y) * width + ox) * 4,
      );
    }
  }
  return { picture: { width, height, rgba }, tiles: { shown, total: channels, columns } };
}

/** A [rows, columns] matrix as a heatmap, shrunk to at most `limit` a side. */
function matrixPicture(
  data: Float32Array,
  rows: number,
  columns: number,
  limit: number,
  lo: number,
  hi: number,
): Picture {
  const height = Math.min(rows, limit);
  const width = Math.min(columns, limit);
  const rgba = new Uint8ClampedArray(width * height * 4);
  const span = hi - lo || 1;
  for (let y = 0; y < height; y++) {
    const sy = Math.min(rows - 1, Math.floor(((y + 0.5) / height) * rows));
    for (let x = 0; x < width; x++) {
      const sx = Math.min(columns - 1, Math.floor(((x + 0.5) / width) * columns));
      heat((data[sy * columns + sx]! - lo) / span, rgba, (y * width + x) * 4);
    }
  }
  return { width, height, rgba };
}

/** Collapses leading axes so every tensor is seen as an image, a token list or a vector. */
function asDisplay(tensor: SimTensor): SimTensor {
  const { shape } = tensor;
  if (shape.length <= 3) return tensor;
  // (items, h, w, c): show the first item.
  const itemShape = shape.slice(-3);
  const size = itemShape.reduce((a, b) => a * b, 1);
  return {
    shape: itemShape,
    data: tensor.data.subarray(0, size),
    illustrative: tensor.illustrative,
  };
}

function firstChannel(tensor: SimTensor | undefined): Picture | null {
  if (!tensor) return null;
  const display = asDisplay(tensor);
  if (display.shape.length !== 3) return null;
  if (display.shape[2]! <= 4) return colorPicture(display, 64);
  const stats = statsOf(display.data);
  return channelPicture(
    display,
    0,
    64,
    stats.min,
    displayMax(histogramOf(display.data, stats), stats),
  );
}

export function summarize(result: NodeResult, isInput: boolean): TensorView {
  const { tensor, extras } = result;
  const display = asDisplay(tensor);
  const stats = statsOf(tensor.data);
  const rank = display.shape.length;
  const kind: TensorView["kind"] = rank === 3 ? "image" : rank === 2 ? "tokens" : "vector";

  const histogram = histogramOf(tensor.data, stats);
  const top = displayMax(histogram, stats);

  let picture: Picture | null = null;
  let tiles: TensorView["tiles"] = null;
  if (kind === "image") {
    if (isInput && display.shape[2]! <= 4) {
      picture = colorPicture(display, 128);
      tiles = { shown: 1, total: 1, columns: 1 };
    } else ({ picture, tiles } = mosaic(display, stats, top));
  } else if (kind === "tokens") {
    picture = matrixPicture(
      display.data,
      display.shape[0]!,
      display.shape[1]!,
      128,
      stats.min,
      top,
    );
  }

  let values: Float32Array | null = null;
  let probabilities = false;
  if (kind === "vector") {
    const count = Math.min(display.data.length, MAX_VALUES);
    values = new Float32Array(count);
    for (let i = 0; i < count; i++)
      values[i] = display.data[Math.floor((i / count) * display.data.length)]!;
    let sum = 0;
    for (let i = 0; i < display.data.length; i++) sum += display.data[i]!;
    probabilities = stats.min >= 0 && Math.abs(sum - 1) < 1e-3 && display.data.length > 1;
  }

  const attention = extras.attention
    ? {
        heads: extras.attention.heads,
        tokens: extras.attention.tokens,
        pictures: Array.from({ length: extras.attention.heads }, (_, head) => {
          const { tokens, data } = extras.attention!;
          const slice = data.subarray(head * tokens * tokens, (head + 1) * tokens * tokens);
          const local = statsOf(slice);
          return matrixPicture(slice, tokens, tokens, 96, local.min, local.max);
        }),
      }
    : null;

  return {
    nodeId: result.nodeId,
    shape: tensor.shape,
    kind,
    illustrative: tensor.illustrative,
    note: result.note,
    ms: result.ms,
    stats,
    histogram,
    picture,
    tiles,
    values,
    probabilities,
    attention,
    kernel: extras.kernel ?? null,
    focus: { input: firstChannel(result.inputs[0]), output: firstChannel(tensor) },
  };
}

/** Every buffer inside a view, for transferring it out of a worker without copying. */
export function transferables(view: TensorView): ArrayBufferLike[] {
  const pictures = [
    view.picture,
    view.focus.input,
    view.focus.output,
    ...(view.attention?.pictures ?? []),
  ];
  const buffers = pictures
    .filter((item): item is Picture => item !== null)
    .map((item) => item.rgba.buffer);
  if (view.values) buffers.push(view.values.buffer);
  if (view.kernel) buffers.push(view.kernel.data.buffer);
  return [...new Set(buffers)];
}
