/**
 * Dev helper: prints what the deterministic analysis sees in a notebook.
 *
 *   pnpm tsx scripts/inspect-notebook.mts fixtures/notebooks/n4-vit.ipynb
 */
import { readFileSync } from "node:fs";
import { extractSequentialModels } from "../src/core/fastpath";
import { buildRuntimeIR } from "../src/core/ir";
import { analyzeNotebook } from "../src/core/notebook/analyze-notebook";
import { parseNotebook } from "../src/core/notebook/parse";
import { formatStaticValue } from "../src/core/python/evaluate";
import { loadPythonParserForNode } from "../src/core/python/test-utils";

const file = process.argv[2];
if (!file) {
  console.error("usage: pnpm tsx scripts/inspect-notebook.mts <notebook.ipynb>");
  process.exit(1);
}

const parser = await loadPythonParserForNode();
const started = performance.now();
const notebook = parseNotebook(readFileSync(file, "utf8"));
const analysis = analyzeNotebook(notebook, parser);
const elapsed = (performance.now() - started).toFixed(0);

console.log(`\n${file}  (${elapsed} ms)  frameworks: ${analysis.frameworks.join(", ") || "none"}`);
for (const cell of analysis.cells) {
  if (cell.role === "markdown") continue;
  const flags = [
    cell.analysis?.hasSyntaxError ? "SYNTAX-ERR" : "",
    cell.summaries.length ? `summary:${cell.summaries.map((s) => s.totals.total).join("/")}` : "",
  ]
    .filter(Boolean)
    .join(" ");
  console.log(
    `  c${String(cell.index).padEnd(3)} ${cell.role.padEnd(8)} ${cell.confidence.padEnd(6)} ` +
      `${cell.title.slice(0, 46).padEnd(46)} ${flags}  [${cell.signals.join("; ").slice(0, 70)}]`,
  );
}
console.log(`  model cells: ${analysis.modelCells.join(", ")}`);
console.log(`  class counts: ${analysis.classCounts.join(", ") || "-"}`);
const constants = Object.entries(analysis.constants)
  .map(([name, value]) => `${name}=${formatStaticValue(value)}`)
  .join("  ");
console.log(`  constants: ${constants.slice(0, 900)}`);

// Fast path: models read from the code with no AI, checked against saved summaries.
const fast = extractSequentialModels(notebook, analysis, parser);
if (fast.wire) {
  const built = buildRuntimeIR(fast.wire, {
    summaries: analysis.cells.flatMap((cell) => cell.summaries),
    provenance: "heuristic",
  });
  if (built.ok) {
    for (const model of built.ir.models) {
      const { totals, rows } = built.ir.verification.perModel[model.id]!;
      const verdict = totals.match === null ? "no saved summary" : totals.match ? "matches summary" : "MISMATCH";
      console.log(
        `  fast path: ${model.name}  ${built.ir.index.topoOrder[model.id]!.length} layers  ` +
          `${totals.computed.total.toLocaleString("en-US")} params  ${verdict}  (${rows.length} rows compared)`,
      );
    }
  } else console.log(`  fast path: invalid IR: ${built.errors.join("; ")}`);
} else console.log("  fast path: no simple layer stack found (needs AI)");
for (const entry of fast.declined) console.log(`  fast path declined c${entry.cell}:${entry.line}  ${entry.reason}`);
