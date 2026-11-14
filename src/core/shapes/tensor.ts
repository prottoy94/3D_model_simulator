/**
 * Tensor shapes as the shape engine sees them. Shapes are kept in the
 * framework's native layout (Keras NHWC, PyTorch NCHW) so the UI shows what the
 * user's code means. Op code must go through the helpers here instead of
 * indexing raw dims.
 */

/** A known size, or a symbolic one: { sym: "B" } batch, { sym: "R" } RoIs. */
export type Dim = number | { sym: string };

export type Layout = "NHWC" | "NCHW" | "NLC" | "NCL" | "NC" | "NL" | "any";

export type TensorShape = { dims: Dim[]; layout: Layout };

export type Framework = "keras" | "pytorch";

export const BATCH: Dim = { sym: "B" };

/** Thrown by catalog shape functions. The engine turns it into a node issue. */
export class ShapeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ShapeError";
  }
}

export function isKnown(dim: Dim | undefined): dim is number {
  return typeof dim === "number";
}

/** Reads a dim that the op needs as a number. */
export function known(dim: Dim | undefined, what: string): number {
  if (typeof dim !== "number") throw new ShapeError(`${what} is not a known size`);
  return dim;
}

export function dimsEqual(a: Dim, b: Dim): boolean {
  if (typeof a === "number" || typeof b === "number") return a === b;
  return a.sym === b.sym;
}

export function shapesEqual(a: TensorShape, b: TensorShape): boolean {
  return a.dims.length === b.dims.length && a.dims.every((dim, i) => dimsEqual(dim, b.dims[i]!));
}

export function dimText(dim: Dim): string {
  return typeof dim === "number" ? String(dim) : dim.sym;
}

/** "(B, 54, 54, 96)" */
export function shapeText(shape: TensorShape): string {
  return `(${shape.dims.map(dimText).join(", ")})`;
}

/** The layout a tensor of this rank most likely has in the given framework. */
export function layoutForRank(rank: number, framework: Framework): Layout {
  if (rank === 2) return "NC";
  if (rank === 3) return "NLC";
  if (rank === 4) return framework === "keras" ? "NHWC" : "NCHW";
  return "any";
}

export type ImageView = { h: number; w: number; c: number };

/** Height, width and channels of an image-like tensor. */
export function image(shape: TensorShape, what = "input"): ImageView {
  const { dims, layout } = shape;
  if (layout === "NHWC" && dims.length === 4) {
    return {
      h: known(dims[1], "height"),
      w: known(dims[2], "width"),
      c: known(dims[3], "channels"),
    };
  }
  if (layout === "NCHW" && dims.length === 4) {
    return {
      h: known(dims[2], "height"),
      w: known(dims[3], "width"),
      c: known(dims[1], "channels"),
    };
  }
  throw new ShapeError(
    `${what} must be an image tensor (batch, height, width, channels), got ${shapeText(shape)}`,
  );
}

export function isImage(shape: TensorShape): boolean {
  return shape.dims.length === 4 && (shape.layout === "NHWC" || shape.layout === "NCHW");
}

/** Same batch and layout as `like`, with a new height, width and channel count. */
export function imageLike(like: TensorShape, h: number, w: number, c: number): TensorShape {
  const batch = like.dims[0] ?? BATCH;
  return like.layout === "NCHW"
    ? { dims: [batch, c, h, w], layout: "NCHW" }
    : { dims: [batch, h, w, c], layout: "NHWC" };
}

/** Index of the channel / feature axis. */
export function featureAxis(shape: TensorShape): number {
  if (shape.layout === "NCHW" || shape.layout === "NCL") return 1;
  return shape.dims.length - 1;
}

/** Size of the channel / feature axis (what BatchNorm and Dense care about). */
export function featureDim(shape: TensorShape, what = "feature size"): number {
  if (shape.dims.length < 2) throw new ShapeError(`${what}: tensor has no feature axis`);
  return known(shape.dims[featureAxis(shape)], what);
}

export function lastDim(shape: TensorShape, what = "last dimension"): number {
  if (shape.dims.length < 2) throw new ShapeError(`${what}: tensor has no feature axis`);
  return known(shape.dims[shape.dims.length - 1], what);
}

/** Product of every known dim after the batch. Symbolic dims count as 1. */
export function elementsPerSample(shape: TensorShape): number {
  return shape.dims.slice(1).reduce<number>((total, dim) => total * (isKnown(dim) ? dim : 1), 1);
}

/** Number of positions the last axis is applied over (tokens, pixels). */
export function positions(shape: TensorShape): number {
  return shape.dims
    .slice(1, -1)
    .reduce<number>((total, dim) => total * (isKnown(dim) ? dim : 1), 1);
}

/** Turns a Python-style axis (negative counts from the end) into an index. */
export function resolveAxis(axis: number, rank: number): number {
  const index = axis < 0 ? rank + axis : axis;
  if (!Number.isInteger(index) || index < 0 || index >= rank) {
    throw new ShapeError(`axis ${axis} is out of range for a tensor with ${rank} dimensions`);
  }
  return index;
}

export function withLastDim(shape: TensorShape, size: number): TensorShape {
  return { dims: [...shape.dims.slice(0, -1), size], layout: shape.layout };
}
