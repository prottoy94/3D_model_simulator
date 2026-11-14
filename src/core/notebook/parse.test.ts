import { describe, expect, it } from "vitest";
import { parseNotebook, toStrippedIpynb } from "./parse";
import { htmlToText, stripAnsi, truncateMiddle } from "./text";
import { NotebookParseError } from "./types";

const ESC = "\u001b";

const sample = JSON.stringify({
  nbformat: 4,
  nbformat_minor: 5,
  metadata: { kernelspec: { name: "tfenv" }, language_info: { name: "python" } },
  cells: [
    { cell_type: "markdown", source: ["# Title\n", "text"] },
    {
      cell_type: "code",
      id: "abc",
      execution_count: 3,
      source: "print('hi')\r\nx = 1",
      outputs: [
        { output_type: "stream", name: "stdout", text: ["hi\n"] },
        {
          output_type: "display_data",
          data: {
            "image/png": "A".repeat(5000),
            "text/plain": ["<Figure size 640x480 with 1 Axes>"],
          },
        },
        {
          output_type: "display_data",
          data: { "text/plain": [`${ESC}[1mModel: "AlexNet"${ESC}[0m`] },
        },
        {
          output_type: "execute_result",
          data: {
            "text/plain": ["<pandas.DataFrame object>"],
            "text/html": [
              "<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2 &amp; 3</td></tr></table>",
            ],
          },
        },
        { output_type: "error", ename: "ValueError", evalue: "bad shape", traceback: ["long"] },
      ],
    },
    { cell_type: "weird", source: "x" },
  ],
});

describe("parseNotebook", () => {
  const notebook = parseNotebook(sample);

  it("reads metadata and keeps original cell indexes", () => {
    expect(notebook.nbformat).toBe("4.5");
    expect(notebook.kernel).toBe("tfenv");
    expect(notebook.cells.map((cell) => cell.index)).toEqual([0, 1]);
    expect(notebook.cells[1].id).toBe("abc");
    expect(notebook.cells[0].id).toBe("cell-0");
  });

  it("joins list sources and normalizes line endings", () => {
    expect(notebook.cells[0].source).toBe("# Title\ntext");
    expect(notebook.cells[1].source).toBe("print('hi')\nx = 1");
  });

  it("drops images and opaque reprs, strips ANSI, converts HTML tables", () => {
    const texts = notebook.cells[1].outputs.map((output) => output.text);
    expect(texts).toEqual(["hi", 'Model: "AlexNet"', "a | b\n1 | 2 & 3", "ValueError: bad shape"]);
    expect(notebook.stats.droppedImages).toBe(1);
    expect(notebook.stats.droppedBytes).toBeGreaterThanOrEqual(5000);
    expect(notebook.stats).toMatchObject({ codeCells: 1, markdownCells: 1 });
  });

  it("round-trips through the stripped ipynb writer", () => {
    const again = parseNotebook(toStrippedIpynb(notebook));
    expect(again.cells.map((cell) => cell.source)).toEqual(
      notebook.cells.map((cell) => cell.source),
    );
    expect(again.cells[1].outputs.slice(0, 3).map((output) => output.text)).toEqual(
      notebook.cells[1].outputs.slice(0, 3).map((output) => output.text),
    );
    expect(again.stats.droppedImages).toBe(0);
  });

  it("rejects files that are not notebooks", () => {
    expect(() => parseNotebook("not json")).toThrow(NotebookParseError);
    expect(() => parseNotebook("{}")).toThrow(/no cells/);
    expect(() => parseNotebook('{"worksheets": []}')).toThrow(/nbformat 3/);
  });
});

describe("text helpers", () => {
  it("strips colors and hyperlinks", () => {
    expect(stripAnsi(`${ESC}[38;5;34m614,656${ESC}[0m`)).toBe("614,656");
    expect(stripAnsi(`${ESC}]8;;http://x\u0007link${ESC}]8;;\u0007`)).toBe("link");
  });
  it("never keeps script content", () => {
    expect(htmlToText("<script>alert(1)</script><p>ok</p>")).toBe("ok");
  });
  it("truncates the middle of long text", () => {
    const cut = truncateMiddle("a".repeat(50) + "b".repeat(50), 20);
    expect(cut.truncated).toBe(true);
    expect(cut.text.startsWith("a".repeat(12))).toBe(true);
    expect(cut.text.endsWith("b".repeat(8))).toBe(true);
    expect(truncateMiddle("short", 20)).toEqual({ text: "short", truncated: false });
  });
});
