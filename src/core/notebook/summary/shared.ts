import type { SummaryDim, SummaryRow, SummaryTotals } from "./types";

/** "46,781,192" → 46781192. Returns null for "--", "(recursive)" and other non-numbers. */
export function parseCount(text: string): number | null {
  const cleaned = text.replace(/[,_\s]/g, "");
  if (!/^\d+$/.test(cleaned)) return null;
  return Number(cleaned);
}

/**
 * "(None, 54, 54, 96)" → [null, 54, 54, 96]; "[1, 6, 28, 28]" → [1, 6, 28, 28].
 * When a layer has several outputs ("[(None, 224, 224, 3)]"), the first one is returned.
 */
export function parseShape(text: string): SummaryDim[] | null {
  const match = text.match(/[([]\s*([^()[\]]*?)\s*[)\]]/);
  if (!match) return null;
  const body = match[1].trim();
  if (body === "") return [];
  const dims: SummaryDim[] = [];
  for (const part of body.split(",")) {
    const token = part.trim();
    if (token === "") continue; // trailing comma, as in "(None,)"
    if (/^\d+$/.test(token)) dims.push(Number(token));
    else if (token === "None" || token === "?" || token === "-1") dims.push(null);
    else return null;
  }
  return dims;
}

const TOTAL_PATTERNS: Array<[keyof SummaryTotals, RegExp]> = [
  ["nonTrainable", /^\s*Non-trainable params:\s*([\d,]+)/im],
  ["trainable", /^\s*Trainable params:\s*([\d,]+)/im],
  ["total", /^\s*Total params:\s*([\d,]+)/im],
];

/** Reads the "Total / Trainable / Non-trainable params" footer that all formats share. */
export function parseTotals(text: string): SummaryTotals {
  const totals: SummaryTotals = { total: null, trainable: null, nonTrainable: null };
  for (const [key, pattern] of TOTAL_PATTERNS) {
    const match = text.match(pattern);
    if (match) totals[key] = parseCount(match[1]);
  }
  return totals;
}

/** "conv2d_1 (Conv2D)" → name + type. Handles names or types cut short with "…". */
export function splitNameAndType(cell: string): Pick<SummaryRow, "name" | "type" | "truncated"> {
  const text = cell.replace(/\s+/g, " ").trim();
  const truncated = text.includes("…");
  const open = text.lastIndexOf("(");
  if (open <= 0) return { name: text.replace(/…/g, ""), type: "", truncated };
  const name = text.slice(0, open).trim().replace(/…/g, "");
  const type = text
    .slice(open + 1)
    .replace(/\)\s*$/, "")
    .replace(/…/g, "")
    .trim();
  return { name, type, truncated };
}
