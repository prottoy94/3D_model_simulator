import { parseCount, parseShape, parseTotals } from "./shared";
import type { ModelSummary, SummaryRow } from "./types";

/**
 * torchinfo (and the older torchsummary) print a tree:
 *
 *   ==========================================================================
 *   Layer (type:depth-idx)                   Output Shape              Param #
 *   ==========================================================================
 *   Net                                      [1, 10]                   --
 *   ├─Conv2d: 1-1                            [1, 6, 28, 28]            156
 *   │    └─ReLU: 2-1                         [1, 6, 28, 28]            --
 *   ==========================================================================
 *   Total params: 61,706
 *
 * Which columns appear is configurable, so cells are located through the header.
 */
export function parseTorchinfoSummaries(text: string): ModelSummary[] {
  const lines = text.split("\n");
  const summaries: ModelSummary[] = [];

  for (let i = 0; i < lines.length; i++) {
    const headerLine = lines[i];
    if (!/^\s*Layer \(type[^)]*\)/.test(headerLine) || !/Param #/.test(headerLine)) continue;
    if (!/^\s*[=-]{10,}\s*$/.test(lines[i + 1] ?? "")) continue;
    if (
      /^\s*Layer \(type\)\s+Output Shape\s+Param #/.test(headerLine) &&
      isKerasRule(lines[i - 1])
    ) {
      continue; // Keras 2 table, handled by its own parser
    }

    const headers = headerLine
      .trim()
      .split(/\s{2,}/)
      .map((cell) => cell.toLowerCase());
    const shapeColumn = headers.findIndex((cell) => cell.startsWith("output shape"));
    const paramColumn = headers.findIndex((cell) => cell.startsWith("param #"));

    const rows: SummaryRow[] = [];
    let modelName: string | null = null;
    let end = i + 2;
    for (; end < lines.length; end++) {
      const line = lines[end];
      if (/^\s*[=-]{10,}\s*$/.test(line)) break;
      if (line.trim() === "") continue;

      const tree = line.match(/^[\s│|├└─+\-']*/)?.[0] ?? "";
      const cells = line
        .slice(tree.length)
        .trim()
        .split(/\s{2,}/);
      const label = cells[0] ?? "";
      const labelMatch = label.match(/^(.+?)(?:\s*\((.+?)\))?:\s*(\d+)-\d+$/);

      if (!labelMatch) {
        // The first row without a "depth-idx" suffix is the model itself.
        if (rows.length === 0 && modelName === null) modelName = label;
        continue;
      }
      const shapeText = shapeColumn > 0 ? (cells[shapeColumn] ?? "") : "";
      rows.push({
        name: labelMatch[2] ?? "",
        type: labelMatch[1].trim(),
        truncated: false,
        outputShape: parseShape(shapeText),
        shapeText,
        params: paramColumn > 0 ? parseCount(cells[paramColumn] ?? "") : null,
        connectedTo: [],
        depth: Number(labelMatch[3]),
      });
    }

    if (rows.length === 0) continue;
    summaries.push({
      format: "torchinfo",
      modelName,
      rows,
      totals: parseTotals(lines.slice(end + 1, end + 8).join("\n")),
    });
    i = end;
  }
  return summaries;
}

function isKerasRule(line: string | undefined): boolean {
  return /^\s*_{10,}\s*$/.test(line ?? "");
}
