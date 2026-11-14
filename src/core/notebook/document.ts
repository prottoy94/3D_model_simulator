import type { CellInfo } from "./analyze-notebook";
import type { Notebook } from "./types";

/** Where one notebook cell sits inside the continuous editor document. */
export type DocumentCell = {
  index: number;
  kind: "code" | "markdown" | "raw";
  /** 1-based editor line of the cell's header comment. */
  headerLine: number;
  /** 1-based editor line of the cell's first source line (code cells only). */
  firstLine: number;
  /** 1-based editor line of the cell's last line (the header itself for markdown). */
  lastLine: number;
};

export type NotebookDocument = {
  text: string;
  cells: DocumentCell[];
};

const RULE = "─".repeat(12);

/**
 * Lays the whole notebook out as one read-only Python document for the editor.
 * Code cells keep their source verbatim under a header comment; markdown cells
 * collapse to a one-line heading. The returned map converts between
 * (cell, line-in-cell), which CodeRefs use, and editor lines.
 */
export function buildNotebookDocument(notebook: Notebook, cells: CellInfo[]): NotebookDocument {
  const lines: string[] = [];
  const map: DocumentCell[] = [];

  notebook.cells.forEach((cell, position) => {
    const info = cells[position];
    const title = info?.title ? ` ${info.title}` : "";
    if (lines.length > 0) lines.push("");

    if (cell.kind !== "code") {
      lines.push(`# ¶${title}`);
      map.push({
        index: cell.index,
        kind: cell.kind,
        headerLine: lines.length,
        firstLine: lines.length,
        lastLine: lines.length,
      });
      return;
    }

    lines.push(`# ${RULE} [${cell.index}] ${info?.role ?? "code"} ${RULE}`);
    const headerLine = lines.length;
    const source = cell.source === "" ? [] : cell.source.split("\n");
    lines.push(...source);
    map.push({
      index: cell.index,
      kind: "code",
      headerLine,
      firstLine: headerLine + 1,
      lastLine: headerLine + source.length,
    });
  });

  return { text: lines.join("\n"), cells: map };
}

/** Editor line → the cell that contains it (the nearest cell above for blank gaps). */
export function cellAtLine(document: NotebookDocument, line: number): DocumentCell | null {
  let found: DocumentCell | null = null;
  for (const cell of document.cells) {
    if (cell.headerLine > line) break;
    found = cell;
  }
  return found;
}

/** (cell, 1-based line in that cell) → editor line. Used to reveal a CodeRef. */
export function editorLineFor(
  document: NotebookDocument,
  cellIndex: number,
  lineInCell = 1,
): number | null {
  const cell = document.cells.find((entry) => entry.index === cellIndex);
  if (!cell) return null;
  if (cell.kind !== "code") return cell.headerLine;
  return Math.min(cell.lastLine, cell.firstLine + Math.max(0, lineInCell - 1));
}
