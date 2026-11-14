import { htmlToText, stripAnsi, truncateMiddle } from "./text";
import {
  NotebookParseError,
  type CellKind,
  type CellOutput,
  type Notebook,
  type NotebookCell,
  type NotebookStats,
  type OutputKind,
} from "./types";

/**
 * Per-output cap. It must stay above the largest model.summary() table we expect
 * (EfficientNetB0 is about 40k characters); only runaway logs get cut.
 */
const MAX_OUTPUT_CHARS = 200_000;

type Json = Record<string, unknown>;

function isObject(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** nbformat stores multi-line strings either as one string or as a list of lines. */
function joinText(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.filter((part) => typeof part === "string").join("");
  return "";
}

function approxBytes(value: unknown): number {
  if (typeof value === "string") return value.length;
  if (Array.isArray(value)) return value.reduce((sum: number, part) => sum + approxBytes(part), 0);
  if (isObject(value)) return JSON.stringify(value).length;
  return 0;
}

/** A text/plain repr such as "<Figure size 640x480 with 1 Axes>" carries no information. */
function isOpaqueRepr(text: string): boolean {
  return /^<[\w.]+( [^>]*)?>$/.test(text.trim());
}

function textFromData(data: Json, stats: NotebookStats): string {
  for (const [mime, payload] of Object.entries(data)) {
    const isText = mime === "text/plain" || mime === "text/html" || mime === "text/markdown";
    if (!isText) {
      if (mime.startsWith("image/")) stats.droppedImages += 1;
      stats.droppedBytes += approxBytes(payload);
    }
  }
  const plain = joinText(data["text/plain"]);
  const markdown = joinText(data["text/markdown"]);
  const html = joinText(data["text/html"]);
  if (markdown) return markdown;
  if (plain && !isOpaqueRepr(plain)) return plain;
  if (html) return htmlToText(html);
  return "";
}

function parseOutput(raw: unknown, stats: NotebookStats): CellOutput | null {
  if (!isObject(raw)) return null;
  let kind: OutputKind;
  let text: string;
  switch (raw.output_type) {
    case "stream":
      kind = "stream";
      text = joinText(raw.text);
      break;
    case "execute_result":
    case "display_data":
      kind = raw.output_type === "execute_result" ? "result" : "display";
      text = isObject(raw.data) ? textFromData(raw.data, stats) : "";
      break;
    case "error":
      kind = "error";
      text = `${joinText(raw.ename)}: ${joinText(raw.evalue)}`;
      break;
    default:
      return null;
  }
  const cleaned = stripAnsi(text).replace(/\r\n?/g, "\n").trimEnd();
  if (cleaned.trim().length === 0) return null;
  const cut = truncateMiddle(cleaned, MAX_OUTPUT_CHARS);
  return { kind, text: cut.text, truncated: cut.truncated };
}

function parseCell(raw: unknown, index: number, stats: NotebookStats): NotebookCell | null {
  if (!isObject(raw)) return null;
  const type = raw.cell_type;
  const kind: CellKind | null =
    type === "code" || type === "markdown" || type === "raw" ? type : null;
  if (!kind) return null;

  const source = joinText(raw.source).replace(/\r\n?/g, "\n");
  const outputs: CellOutput[] = [];
  if (kind === "code" && Array.isArray(raw.outputs)) {
    for (const rawOutput of raw.outputs) {
      const output = parseOutput(rawOutput, stats);
      if (output) outputs.push(output);
    }
  }
  if (kind === "code") stats.codeCells += 1;
  if (kind === "markdown") stats.markdownCells += 1;

  // Markdown cells can embed images as attachments.
  if (isObject(raw.attachments)) stats.droppedBytes += approxBytes(raw.attachments);

  return {
    index,
    id: typeof raw.id === "string" && raw.id ? raw.id : `cell-${index}`,
    kind,
    source,
    outputs,
    executionCount: typeof raw.execution_count === "number" ? raw.execution_count : null,
  };
}

/**
 * Parses the text of an .ipynb file into a text-only Notebook.
 * Images, widgets and raw HTML are dropped here, before anything else sees them.
 */
export function parseNotebook(fileText: string): Notebook {
  let root: unknown;
  try {
    root = JSON.parse(fileText);
  } catch {
    throw new NotebookParseError("This file is not valid JSON, so it is not a Jupyter notebook.");
  }
  if (!isObject(root) || !Array.isArray(root.cells)) {
    if (isObject(root) && Array.isArray(root.worksheets)) {
      throw new NotebookParseError(
        "This notebook uses the old nbformat 3. Open and re-save it in Jupyter to upgrade it.",
      );
    }
    throw new NotebookParseError("This file has no cells, so it is not a Jupyter notebook.");
  }

  const metadata = isObject(root.metadata) ? root.metadata : {};
  const kernelspec = isObject(metadata.kernelspec) ? metadata.kernelspec : {};
  const languageInfo = isObject(metadata.language_info) ? metadata.language_info : {};
  const stats: NotebookStats = {
    codeCells: 0,
    markdownCells: 0,
    droppedImages: 0,
    droppedBytes: 0,
  };

  const cells: NotebookCell[] = [];
  root.cells.forEach((rawCell, index) => {
    const cell = parseCell(rawCell, index, stats);
    if (cell) cells.push(cell);
  });

  const major = typeof root.nbformat === "number" ? root.nbformat : "?";
  const minor = typeof root.nbformat_minor === "number" ? root.nbformat_minor : 0;
  return {
    nbformat: `${major}.${minor}`,
    language:
      (typeof languageInfo.name === "string" && languageInfo.name) ||
      (typeof kernelspec.language === "string" && kernelspec.language) ||
      "python",
    kernel: typeof kernelspec.name === "string" ? kernelspec.name : null,
    cells,
    stats,
  };
}

/**
 * Writes a Notebook back as a small, valid .ipynb (text outputs only).
 * Used to create stripped test fixtures from large real notebooks.
 */
export function toStrippedIpynb(notebook: Notebook): string {
  const [major, minor] = notebook.nbformat.split(".").map((part) => Number(part) || 0);
  const cells = notebook.cells.map((cell) => {
    const base = { cell_type: cell.kind, id: cell.id, metadata: {}, source: cell.source };
    if (cell.kind !== "code") return base;
    return {
      ...base,
      execution_count: cell.executionCount,
      outputs: cell.outputs.map((output) => {
        if (output.kind === "stream") {
          return { output_type: "stream", name: "stdout", text: output.text };
        }
        if (output.kind === "error") {
          return { output_type: "error", ename: output.text, evalue: "", traceback: [] };
        }
        return {
          output_type: output.kind === "result" ? "execute_result" : "display_data",
          data: { "text/plain": output.text },
          metadata: {},
          ...(output.kind === "result" ? { execution_count: cell.executionCount } : {}),
        };
      }),
    };
  });
  return JSON.stringify(
    {
      cells,
      metadata: {
        kernelspec: { name: notebook.kernel ?? "python3", language: notebook.language },
        language_info: { name: notebook.language },
      },
      nbformat: major || 4,
      nbformat_minor: minor,
    },
    null,
    1,
  );
}
