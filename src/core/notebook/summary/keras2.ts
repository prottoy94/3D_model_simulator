import { parseCount, parseShape, parseTotals, splitNameAndType } from "./shared";
import type { ModelSummary, SummaryRow } from "./types";

/**
 * Keras 2 (TensorFlow ≤ 2.15) prints a plain-text table:
 *
 *   Model: "sequential"
 *   _________________________________________________________________
 *    Layer (type)                Output Shape              Param #
 *   =================================================================
 *    conv2d (Conv2D)             (None, 54, 54, 96)        34944
 *   =================================================================
 *   Total params: 46,781,192
 *
 * Columns are aligned by position, so cells are cut at the header's column offsets.
 * A long name can push a row onto a continuation line.
 */
export function parseKeras2Summaries(text: string): ModelSummary[] {
  const lines = text.split("\n");
  const summaries: ModelSummary[] = [];

  for (let i = 0; i < lines.length; i++) {
    const headerLine = lines[i];
    if (!/^\s*Layer \(type\)\s+Output Shape\s+Param #/.test(headerLine)) continue;
    if (!/^\s*=+\s*$/.test(lines[i + 1] ?? "")) continue;

    const offsets = {
      shape: headerLine.indexOf("Output Shape"),
      params: headerLine.indexOf("Param #"),
      connected: headerLine.indexOf("Connected to"),
    };
    const modelName = findModelName(lines, i);
    const rows: SummaryRow[] = [];
    let current: { layer: string; shape: string; params: string; connected: string[] } | null =
      null;
    const flush = () => {
      if (!current) return;
      const shapeText = current.shape.replace(/\s+/g, " ").trim();
      rows.push({
        ...splitNameAndType(current.layer),
        outputShape: parseShape(shapeText),
        shapeText,
        params: parseCount(current.params),
        connectedTo: current.connected,
        depth: 0,
      });
      current = null;
    };

    let end = i + 2;
    for (; end < lines.length; end++) {
      const line = lines[end];
      if (/^\s*=+\s*$/.test(line)) break;
      if (/^\s*_+\s*$/.test(line) || line.trim() === "") {
        continue;
      }
      const cut = (from: number, to: number) => line.slice(from, to < 0 ? undefined : to).trim();
      const layer = cut(0, offsets.shape);
      const shape = cut(offsets.shape, offsets.params);
      const params = cut(offsets.params, offsets.connected);
      const connected = offsets.connected < 0 ? "" : cut(offsets.connected, -1);
      const link = connected.replace(/^\[?'?|'?\]?,?$/g, "").replace(/(\[\d+\])+$/, "");

      // A new row always carries a parameter count; anything else continues the previous row.
      if (params !== "" && parseCount(params) !== null) {
        flush();
        current = { layer, shape, params, connected: link ? [link] : [] };
      } else if (current) {
        current.layer = `${current.layer} ${layer}`.trim();
        current.shape = `${current.shape} ${shape}`.trim();
        if (link) current.connected.push(link);
      }
    }
    flush();

    summaries.push({
      format: "keras2",
      modelName,
      rows,
      totals: parseTotals(lines.slice(end + 1, end + 8).join("\n")),
    });
    i = end;
  }
  return summaries;
}

function findModelName(lines: string[], headerIndex: number): string | null {
  for (let j = headerIndex - 1; j >= Math.max(0, headerIndex - 3); j--) {
    const match = lines[j].match(/Model:\s*"([^"]*)"/);
    if (match) return match[1];
  }
  return null;
}
