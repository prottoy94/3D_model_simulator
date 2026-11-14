import { parseCount, parseShape, parseTotals, splitNameAndType } from "./shared";
import type { ModelSummary, SummaryRow } from "./types";

/**
 * Keras 3 prints summaries as a box-drawing table:
 *
 *   ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┓
 *   ┃ Layer (type) ┃ Output Shape ┃ Param # ┃
 *   ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━┩
 *   │ batch_normalization │ (None, 54, 54, 96) │ 384 │     <- one layer can span
 *   │ (BatchNormalization)│                    │     │        several physical lines
 *   ├──────────────┼──────────────┼─────────┤
 *
 * Long names wrap onto extra lines or are cut with "…". ANSI codes are already
 * stripped by the notebook parser.
 */
export function parseKeras3Summaries(text: string): ModelSummary[] {
  const lines = text.split("\n");
  const summaries: ModelSummary[] = [];

  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].trimStart().startsWith("┏")) continue;

    const modelName = findModelName(lines, i);
    const header = splitRow(lines[i + 1] ?? "", "┃").map((cell) => cell.toLowerCase());
    const column = {
      layer: header.findIndex((cell) => cell.startsWith("layer")),
      shape: header.findIndex((cell) => cell.includes("shape")),
      params: header.findIndex((cell) => cell.includes("param")),
      connected: header.findIndex((cell) => cell.includes("connected")),
    };
    if (column.layer < 0) continue;

    const rows: SummaryRow[] = [];
    let pending: string[][] = [];
    const flush = () => {
      if (pending.length > 0) rows.push(buildRow(pending, column));
      pending = [];
    };

    let end = i + 2;
    for (; end < lines.length; end++) {
      const line = lines[end].trimStart();
      if (line.startsWith("│")) pending.push(splitRow(line, "│"));
      else if (line.startsWith("├") || line.startsWith("┡")) flush();
      else if (line.startsWith("└")) {
        flush();
        break;
      } else if (line.trim() !== "") break; // table ended without a closing border
    }
    flush();

    // The totals footer follows the table, before any next "Model:" block.
    const footer: string[] = [];
    for (let j = end + 1; j < lines.length && footer.length < 8; j++) {
      if (lines[j].trimStart().startsWith("┏") || /^\s*Model:/.test(lines[j])) break;
      footer.push(lines[j]);
    }

    summaries.push({
      format: "keras3",
      modelName,
      rows,
      totals: parseTotals(footer.join("\n")),
    });
    i = end;
  }
  return summaries;
}

function findModelName(lines: string[], tableStart: number): string | null {
  for (let j = tableStart - 1; j >= Math.max(0, tableStart - 3); j--) {
    const match = lines[j].match(/Model:\s*"([^"]*)"/);
    if (match) return match[1];
  }
  return null;
}

function splitRow(line: string, bar: string): string[] {
  const parts = line.split(bar);
  return parts.slice(1, -1).map((cell) => cell.trim());
}

type Columns = { layer: number; shape: number; params: number; connected: number };

function buildRow(physicalLines: string[][], column: Columns): SummaryRow {
  const pieces = (index: number): string[] =>
    index < 0 ? [] : physicalLines.map((cells) => cells[index] ?? "").filter((cell) => cell !== "");

  const shapeText = pieces(column.shape).join(" ").replace(/\s+/g, " ");
  const connectedTo = pieces(column.connected)
    .filter((cell) => cell !== "-")
    // "conv2d[0][0]" → "conv2d"; a cut-off "conv2d[0]…" keeps its prefix.
    .map((cell) => cell.replace(/(\[\d+\])+…?$/, "").replace(/…$/, ""));

  return {
    ...splitNameAndType(pieces(column.layer).join(" ")),
    outputShape: parseShape(shapeText),
    shapeText,
    params: parseCount(pieces(column.params).join("")),
    connectedTo,
    depth: 0,
  };
}
