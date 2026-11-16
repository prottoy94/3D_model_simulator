/**
 * Box geometry for the detection steps. Kept apart from the kernels so it can be
 * imported on its own (kernels.ts and kernels-extra.ts import each other).
 */

/** Intersection over union of two boxes given as [x1, y1, x2, y2]. */
export function iou(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const width = Math.min(a[2]!, b[2]!) - Math.max(a[0]!, b[0]!);
  const height = Math.min(a[3]!, b[3]!) - Math.max(a[1]!, b[1]!);
  if (width <= 0 || height <= 0) return 0;
  const overlap = width * height;
  const areaA = (a[2]! - a[0]!) * (a[3]! - a[1]!);
  const areaB = (b[2]! - b[0]!) * (b[3]! - b[1]!);
  return overlap / (areaA + areaB - overlap);
}

/** Greedy non-maximum suppression. Returns the indices of the kept boxes, best first. */
export function suppress(
  boxes: Float32Array,
  scores: ArrayLike<number>,
  iouThreshold: number,
  scoreThreshold: number,
  maxOutput: number,
): number[] {
  const order = Array.from({ length: scores.length }, (_, i) => i)
    .filter((i) => scores[i]! >= scoreThreshold)
    .sort((a, b) => scores[b]! - scores[a]! || a - b);
  const kept: number[] = [];
  for (const candidate of order) {
    if (kept.length >= maxOutput) break;
    const box = boxes.subarray(candidate * 4, candidate * 4 + 4);
    if (kept.every((other) => iou(box, boxes.subarray(other * 4, other * 4 + 4)) <= iouThreshold)) {
      kept.push(candidate);
    }
  }
  return kept;
}
