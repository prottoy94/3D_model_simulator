/**
 * Runs the AI extraction on notebooks and reports whether the engine could
 * verify the result against each notebook's saved summaries.
 *
 *   pnpm eval:ai <provider> <model> <notebook.ipynb> [more.ipynb ...]
 *
 * It talks to the running dev server (pnpm dev), so API keys stay there.
 * THIS SENDS THE NOTEBOOKS' MODEL CODE TO THE PROVIDER and uses free-tier
 * quota. Only run it on notebooks you are happy to send.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { buildExtractionContext } from "../src/core/ai/context";
import { draftToWire } from "../src/core/ai/draft";
import type { ExtractEvent } from "../src/core/ai/events";
import { buildRuntimeIR } from "../src/core/ir/build";
import { analyzeNotebook } from "../src/core/notebook/analyze-notebook";
import { parseNotebook } from "../src/core/notebook/parse";
import { loadPythonParserForNode } from "../src/core/python/test-utils";

const [provider, model, ...files] = process.argv.slice(2);
if (!provider || !model || files.length === 0) {
  console.error("usage: pnpm eval:ai <provider> <model> <notebook.ipynb> [more.ipynb ...]");
  process.exit(1);
}
const server = process.env.NEUROSCOPE_URL ?? "http://localhost:3000";
const parser = await loadPythonParserForNode();
const count = (value: number) => value.toLocaleString("en-US");
let failures = 0;

for (const file of files) {
  const notebook = parseNotebook(readFileSync(file, "utf8"));
  const analysis = analyzeNotebook(notebook, parser);
  const context = buildExtractionContext(notebook, analysis);
  const summaries = analysis.cells.flatMap((cell) => cell.summaries);
  console.log(`\n== ${path.basename(file)}  cells ${context.cells.join(", ")}  ~${count(context.approxTokens)} tokens`);

  const started = Date.now();
  const response = await fetch(`${server}/api/ai/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, model, context: context.text, summaries }),
  });
  const events = (await response.text())
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as ExtractEvent);
  const seconds = ((Date.now() - started) / 1000).toFixed(1);

  for (const event of events) {
    if (event.type === "stage" && event.stage === "repair") {
      console.log(`   repair: ${event.message}`);
      for (const detail of event.details ?? []) console.log(`     - ${detail}`);
    }
  }
  const error = events.find((event) => event.type === "error");
  const result = events.find((event) => event.type === "result");
  if (error?.type === "error" || result?.type !== "result") {
    failures++;
    console.log(`   FAILED in ${seconds}s: ${error?.type === "error" ? error.message : "no result"}`);
    continue;
  }

  const built = buildRuntimeIR(draftToWire(result.draft), { summaries, provenance: "ai" });
  if (!built.ok) {
    failures++;
    console.log(`   INVALID IR: ${built.errors.join("; ")}`);
    continue;
  }
  for (const entry of built.ir.models) {
    const totals = built.ir.verification.perModel[entry.id]!.totals;
    const verdict = totals.match === null ? "no summary" : totals.match ? "MATCH" : "MISMATCH";
    if (totals.match === false) failures++;
    const nodes = built.ir.index.topoOrder[entry.id]!;
    const flagged = nodes.filter((id) => ["error", "mismatch"].includes(built.ir.nodes[id]!.status)).length;
    const linked = nodes.filter((id) => built.ir.nodes[id]!.refs.length > 0).length;
    console.log(
      `   ${entry.name}: ${nodes.length} layers, ${count(totals.computed.total)} params ` +
        `(trainable ${count(totals.computed.trainable)})  ${verdict}  flagged ${flagged}  with code lines ${linked}/${nodes.length}`,
    );
  }
  const usage = result.usage ? `${count(result.usage.input)} in / ${count(result.usage.output)} out tokens` : "usage not reported";
  console.log(`   ${seconds}s, ${result.rounds} repair round(s), ${usage}, unresolved: ${result.problems.length}`);
}

process.exit(failures > 0 ? 1 : 0);
