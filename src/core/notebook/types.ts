export type CellKind = "code" | "markdown" | "raw";

export type OutputKind = "stream" | "result" | "display" | "error";

/** A text-only view of one notebook output. Images, widgets and raw HTML never get this far. */
export type CellOutput = {
  kind: OutputKind;
  text: string;
  /** True when the middle of a very long output was cut out. */
  truncated: boolean;
};

export type NotebookCell = {
  /** Position in the original file (0-based). Every CodeRef uses this index. */
  index: number;
  id: string;
  kind: CellKind;
  source: string;
  outputs: CellOutput[];
  executionCount: number | null;
};

export type NotebookStats = {
  codeCells: number;
  markdownCells: number;
  droppedImages: number;
  /** Approximate size of the dropped binary payloads, in bytes. */
  droppedBytes: number;
};

export type Notebook = {
  nbformat: string;
  language: string;
  kernel: string | null;
  cells: NotebookCell[];
  stats: NotebookStats;
};

export class NotebookParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NotebookParseError";
  }
}
