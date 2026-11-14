import { describe, expect, it } from "vitest";
import type { CellInfo } from "./analyze-notebook";
import { buildNotebookDocument, cellAtLine, editorLineFor } from "./document";
import { parseNotebook } from "./parse";

const notebook = parseNotebook(
  JSON.stringify({
    nbformat: 4,
    nbformat_minor: 5,
    cells: [
      { cell_type: "markdown", source: "# Intro" },
      { cell_type: "code", source: "a = 1\nb = 2", outputs: [] },
      { cell_type: "markdown", source: "## Model" },
      { cell_type: "code", source: "x = Dense(4)", outputs: [] },
    ],
  }),
);

const info = (index: number, role: CellInfo["role"], title: string): CellInfo => ({
  index,
  role,
  title,
  confidence: "high",
  signals: [],
  lineCount: 1,
  analysis: null,
  summaries: [],
  facts: { classCounts: [], paramMentions: [] },
});

const document = buildNotebookDocument(notebook, [
  info(0, "markdown", "Intro"),
  info(1, "config", "a = 1"),
  info(2, "markdown", "Model"),
  info(3, "model", "x = Dense(4)"),
]);

describe("buildNotebookDocument", () => {
  it("keeps code verbatim under headers and collapses markdown", () => {
    expect(document.text.split("\n")).toEqual([
      "# ¶ Intro",
      "",
      "# ──────────── [1] config ────────────",
      "a = 1",
      "b = 2",
      "",
      "# ¶ Model",
      "",
      "# ──────────── [3] model ────────────",
      "x = Dense(4)",
    ]);
  });

  it("maps cells to editor lines", () => {
    expect(document.cells[1]).toEqual({
      index: 1,
      kind: "code",
      headerLine: 3,
      firstLine: 4,
      lastLine: 5,
    });
    expect(editorLineFor(document, 1, 2)).toBe(5);
    expect(editorLineFor(document, 3)).toBe(10);
    expect(editorLineFor(document, 2)).toBe(7);
    expect(editorLineFor(document, 1, 99)).toBe(5);
    expect(editorLineFor(document, 42)).toBeNull();
  });

  it("maps editor lines back to cells", () => {
    expect(cellAtLine(document, 1)?.index).toBe(0);
    expect(cellAtLine(document, 5)?.index).toBe(1);
    expect(cellAtLine(document, 6)?.index).toBe(1);
    expect(cellAtLine(document, 10)?.index).toBe(3);
  });
});
