import type { Tree } from "web-tree-sitter";
import { analyzeCell, type CellAnalysis, type PySymbol } from "../python/analyze";
import type { StaticValue } from "../python/evaluate";
import { sanitizePython } from "../python/sanitize";
import { classifyCodeCell, detectFrameworks, type CellRole, type RoleResult } from "./roles";
import { extractOutputFacts, parseSummaries, type ModelSummary, type OutputFacts } from "./summary";
import type { Notebook } from "./types";

/** Anything that can turn Python source into a tree-sitter tree. */
export type PythonParser = { parse(source: string): Tree | null };

export type CellInfo = RoleResult & {
  index: number;
  /** Short label for the Notebook Map: the first heading, definition or line. */
  title: string;
  lineCount: number;
  analysis: CellAnalysis | null;
  summaries: ModelSummary[];
  facts: OutputFacts;
};

export type NotebookAnalysis = {
  /** One entry per notebook cell, in the same order as `notebook.cells`. */
  cells: CellInfo[];
  /** Module-level definitions across the notebook, in order. */
  symbols: PySymbol[];
  /** Constants whose values were worked out statically, e.g. NUM_PATCHES → 196. */
  constants: Record<string, StaticValue>;
  frameworks: string[];
  /** Original indexes of the cells that define architecture. */
  modelCells: number[];
  /** Class counts read from outputs, used to resolve `num_classes`. */
  classCounts: number[];
};

const NO_FACTS: OutputFacts = { classCounts: [], paramMentions: [] };

/**
 * Stage 3 of the pipeline ("Locate"): tags every cell with a role, builds the
 * symbol table and reads the facts printed in outputs. Deterministic, no AI.
 */
export function analyzeNotebook(notebook: Notebook, parser: PythonParser): NotebookAnalysis {
  const scope = new Map<string, StaticValue>();
  const symbols: PySymbol[] = [];
  const importedModules: string[] = [];
  const classCounts: number[] = [];

  const cells = notebook.cells.map((cell): CellInfo => {
    const lineCount = cell.source === "" ? 0 : cell.source.split("\n").length;
    if (cell.kind !== "code") {
      return {
        index: cell.index,
        role: cell.kind === "markdown" ? "markdown" : "other",
        confidence: "high",
        signals: [],
        title: markdownTitle(cell.source),
        lineCount,
        analysis: null,
        summaries: [],
        facts: NO_FACTS,
      };
    }

    const tree = parser.parse(sanitizePython(cell.source));
    const analysis = tree ? analyzeCell(tree, cell.index, scope) : null;
    tree?.delete();

    const outputText = cell.outputs.map((output) => output.text).join("\n");
    const facts = outputText ? extractOutputFacts(outputText) : NO_FACTS;
    for (const count of facts.classCounts) {
      if (!classCounts.includes(count)) classCounts.push(count);
    }
    if (analysis) {
      symbols.push(...analysis.defines);
      importedModules.push(...analysis.importedModules);
    }

    const role: RoleResult = analysis
      ? classifyCodeCell(analysis, cell.source)
      : { role: "other", confidence: "low", signals: ["could not be parsed"] };

    return {
      index: cell.index,
      ...role,
      title: codeTitle(cell.source, analysis, role.role),
      lineCount,
      analysis,
      summaries: outputText ? parseSummaries(outputText) : [],
      facts,
    };
  });

  return {
    cells,
    symbols,
    constants: Object.fromEntries(scope),
    frameworks: detectFrameworks(importedModules),
    modelCells: cells.filter((cell) => cell.role === "model").map((cell) => cell.index),
    classCounts,
  };
}

function markdownTitle(source: string): string {
  const firstLine = source
    .split("\n")
    .find((line) => line.trim() !== "" && !/^-{3,}$/.test(line.trim()));
  return (firstLine ?? "")
    .replace(/^#+\s*/, "")
    .replace(/[*_`]/g, "")
    .trim()
    .slice(0, 80);
}

function codeTitle(source: string, analysis: CellAnalysis | null, role: CellRole): string {
  const defined = (analysis?.defines ?? []).filter(
    (symbol) => symbol.kind === "class" || symbol.kind === "function",
  );
  if (defined.length > 0 && (role === "model" || role === "other" || role === "data")) {
    const names = defined.slice(0, 3).map((symbol) => symbol.name);
    return names.join(", ") + (defined.length > 3 ? ` +${defined.length - 3}` : "");
  }
  // Otherwise the first comment or line of code says what the cell is about.
  const firstLine = source.split("\n").find((line) => /[A-Za-z]/.test(line)) ?? "";
  return firstLine
    .replace(/^\s*#+\s*/, "")
    .replace(/^[=\-─\s]+|[=\-─\s]+$/g, "")
    .trim()
    .slice(0, 80);
}
