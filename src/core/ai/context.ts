/**
 * The context slicer: picks the parts of a notebook an LLM needs to describe
 * its architecture, and nothing else. Only code that defines or configures
 * models is sent, with line numbers, plus the constants and saved summaries
 * the static analysis already found. Images and other outputs never leave.
 */
import type { RuntimeIR } from "../ir/runtime";
import type { NotebookAnalysis } from "../notebook/analyze-notebook";
import type { ModelSummary } from "../notebook/summary";
import type { Notebook } from "../notebook/types";
import { formatStaticValue } from "../python/evaluate";

export type ExtractionContext = {
  text: string;
  /** Indexes of the cells whose source is included. */
  cells: number[];
  chars: number;
  /** A rough estimate (4 characters per token), for the "what will be sent" notice. */
  approxTokens: number;
  truncated: boolean;
};

const DEFAULT_MAX_CHARS = 120_000;
const SUMMARY_HEAD_ROWS = 12;
const SUMMARY_TAIL_ROWS = 14;
const MAX_CONSTANTS = 80;

const count = (value: number | null) => (value === null ? "?" : value.toLocaleString("en-US"));

function summaryText(summary: ModelSummary, cell: number): string {
  const rows = summary.rows.map(
    (row) =>
      `  ${row.name || "-"} (${row.type}) ${row.shapeText || "?"} params=${count(row.params)}`,
  );
  // Long tables (a pretrained backbone has hundreds of rows) keep their start and end only.
  const shown =
    rows.length > SUMMARY_HEAD_ROWS + SUMMARY_TAIL_ROWS + 2
      ? [
          ...rows.slice(0, SUMMARY_HEAD_ROWS),
          `  … ${rows.length - SUMMARY_HEAD_ROWS - SUMMARY_TAIL_ROWS} rows omitted …`,
          ...rows.slice(-SUMMARY_TAIL_ROWS),
        ]
      : rows;
  const { total, trainable, nonTrainable } = summary.totals;
  return [
    `Summary printed by cell ${cell}${summary.modelName ? ` for model "${summary.modelName}"` : ""}: total ${count(total)}, trainable ${count(trainable)}, non-trainable ${count(nonTrainable)}, ${rows.length} rows`,
    ...shown,
  ].join("\n");
}

export function buildExtractionContext(
  notebook: Notebook,
  analysis: NotebookAnalysis,
  maxChars = DEFAULT_MAX_CHARS,
): ExtractionContext {
  const parts: string[] = [];
  parts.push(`FRAMEWORKS: ${analysis.frameworks.join(", ") || "unknown"}`);

  const constants = Object.entries(analysis.constants).slice(0, MAX_CONSTANTS);
  if (constants.length > 0) {
    parts.push(
      "CONSTANTS (worked out from the code):\n" +
        constants
          .map(([name, value]) => `  ${name} = ${formatStaticValue(value).slice(0, 120)}`)
          .join("\n"),
    );
  }
  if (analysis.classCounts.length > 0) {
    parts.push(`NUMBER OF CLASSES reported in cell outputs: ${analysis.classCounts.join(", ")}`);
  }

  const summaries = analysis.cells.flatMap((cell) =>
    cell.summaries.map((summary) => summaryText(summary, cell.index)),
  );
  if (summaries.length > 0)
    parts.push(
      "SAVED MODEL SUMMARIES (ground truth from the notebook's own outputs):\n" +
        summaries.join("\n\n"),
    );

  // Model cells define the layers; setup cells say what is frozen or fine-tuned.
  const wanted = analysis.cells.filter((cell) => cell.role === "model" || cell.role === "compile");
  const included: number[] = [];
  let truncated = false;
  let used = parts.join("\n\n").length;
  const code: string[] = [];
  // Model cells go first when space is short, but the text keeps notebook order.
  const order = [...wanted].sort(
    (a, b) => Number(b.role === "model") - Number(a.role === "model") || a.index - b.index,
  );
  const chosen = new Set<number>();
  for (const cell of order) {
    const source = notebook.cells.find((entry) => entry.index === cell.index)?.source ?? "";
    if (used + source.length + 200 > maxChars) {
      truncated = true;
      continue;
    }
    used += source.length + 200;
    chosen.add(cell.index);
  }
  for (const cell of wanted) {
    if (!chosen.has(cell.index)) continue;
    const source = notebook.cells.find((entry) => entry.index === cell.index)?.source ?? "";
    const numbered = source
      .split("\n")
      .map((line, index) => `${String(index + 1).padStart(4)}| ${line}`)
      .join("\n");
    code.push(
      `### cell ${cell.index} [${cell.role === "model" ? "model" : "training setup"}]\n${numbered}`,
    );
    included.push(cell.index);
  }
  parts.push(
    'CODE (cell numbers and line numbers are the ones to use in "lines"):\n\n' + code.join("\n\n"),
  );
  if (truncated) parts.push("NOTE: some cells were left out because the notebook is very long.");

  const text = parts.join("\n\n");
  return {
    text,
    cells: included,
    chars: text.length,
    approxTokens: Math.ceil(text.length / 4),
    truncated,
  };
}

/**
 * What the engine found wrong, in words an LLM can act on. Empty means the IR
 * is as good as the engine can tell. Declared layers and ops outside the
 * catalog are not problems the model can fix, so they are not listed.
 */
export function describeProblems(ir: RuntimeIR, limit = 14, requireLines = true): string[] {
  const problems: string[] = [];
  for (const issue of ir.issues) if (issue.severity === "error") problems.push(issue.message);

  for (const model of ir.models) {
    const report = ir.verification.perModel[model.id];
    const nodes = (ir.index.topoOrder[model.id] ?? []).map((id) => ir.nodes[id]!);
    for (const node of nodes) {
      if (node.templateId && !node.id.endsWith("#1")) continue; // one report per repeated layer
      const id = node.templateId ?? node.id;
      // A layer that fails only because its input failed is not a separate problem.
      const upstreamFailed = (ir.index.inEdges[node.id] ?? []).some((edge) => {
        const source = ir.nodes[edge.from];
        return source !== undefined && (source.status === "error" || source.outShape === null);
      });
      if (node.status === "error" && upstreamFailed) continue;
      if (node.status === "error") {
        const reasons = node.issues
          .filter((issue) => issue.severity === "error")
          .map((issue) => issue.message);
        problems.push(
          `node "${id.replace(/@\d+$/, "")}" (${node.op}): ${reasons.join("; ") || "could not be computed"}`,
        );
      } else if (node.status === "mismatch") {
        const reason = node.issues.find((issue) => issue.code === "summary-mismatch")?.message;
        problems.push(
          `node "${id.replace(/@\d+$/, "")}" (${node.op}): ${reason ?? "differs from the saved summary"}`,
        );
      }
    }
    const withoutLines = nodes.filter((node) => node.refs.length === 0 && !node.templateId).length;
    if (requireLines && nodes.length > 0 && withoutLines > nodes.length / 2) {
      problems.push(
        `model "${model.name}": ${withoutLines} nodes have no "lines". Add "lines": [cell, firstLine, lastLine] to every node (send only id and lines for each).`,
      );
    }
    if (report?.totals.match === false && report.totals.summary) {
      const { computed, summary } = report.totals;
      problems.push(
        `model "${model.name}": the engine computes ${count(computed.total)} params (trainable ${count(computed.trainable)}, non-trainable ${count(computed.nonTrainable)}) but the saved summary says total ${count(summary.total)}, trainable ${count(summary.trainable)}, non-trainable ${count(summary.nonTrainable)}. Find the missing, extra or wrongly configured or wrongly frozen layers.`,
      );
      for (const row of report.unmatchedRows.slice(0, 6)) {
        problems.push(
          `summary row "${row.name}" (${row.type}, ${count(row.params)} params) has no matching node`,
        );
      }
    }
  }
  return problems.slice(0, limit);
}
