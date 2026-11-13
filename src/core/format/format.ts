/** Display helpers for the numbers the simulator shows everywhere. Pure functions. */

export type DisplayDim = number | string;

/** 46781192 → "46,781,192" */
export function formatCount(value: number): string {
  if (!Number.isFinite(value)) return "–";
  return Math.round(value).toLocaleString("en-US");
}

/** 46781192 → "46.78 M". Used where the exact count would not fit. */
export function formatCompact(value: number): string {
  if (!Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  const units: Array<[number, string]> = [
    [1e12, "T"],
    [1e9, "G"],
    [1e6, "M"],
    [1e3, "K"],
  ];
  for (const [size, suffix] of units) {
    if (abs >= size) return `${trimZeros((value / size).toFixed(2))} ${suffix}`;
  }
  return String(Math.round(value));
}

/** [54, 54, 96] → "54×54×96". Symbolic dims such as "B" pass through. */
export function formatShape(dims: readonly DisplayDim[]): string {
  if (dims.length === 0) return "scalar";
  return dims.join("×");
}

function trimZeros(text: string): string {
  return text.replace(/\.?0+$/, "");
}
