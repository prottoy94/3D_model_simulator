"use client";

import Editor, { type BeforeMount, type OnMount } from "@monaco-editor/react";
import { useEffect, useRef } from "react";
import { cellAtLine, editorLineFor } from "@/core/notebook/document";
import { readToken } from "@/lib/tokens";
import { useProjectStore } from "@/stores/project-store";

type MonacoEditor = Parameters<OnMount>[0];
type Monaco = Parameters<OnMount>[1];

const THEME = "neuroscope";

/**
 * The whole notebook as one read-only Python document (see buildNotebookDocument).
 * Selecting a cell in the Notebook Map scrolls here; moving the cursor here
 * selects the cell in the map.
 */
export function CodeEditor() {
  const document = useProjectStore((state) => state.project?.document);
  const cells = useProjectStore((state) => state.project?.analysis.cells);
  const selectedCell = useProjectStore((state) => state.selectedCell);
  const editorRef = useRef<MonacoEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const decorationsRef = useRef<ReturnType<MonacoEditor["createDecorationsCollection"]> | null>(
    null,
  );

  const defineTheme: BeforeMount = (monaco) => {
    const hex = (name: `--${string}`) => readToken(name).replace("#", "");
    monaco.editor.defineTheme(THEME, {
      base: "vs-dark",
      inherit: true,
      rules: [
        { token: "comment", foreground: hex("--text-3") },
        { token: "keyword", foreground: hex("--op-attn") },
        { token: "string", foreground: hex("--op-act") },
        { token: "number", foreground: hex("--op-linear") },
        { token: "type", foreground: hex("--op-recur") },
        { token: "identifier", foreground: hex("--text-1") },
      ],
      colors: {
        "editor.background": "#00000000",
        "editor.foreground": readToken("--text-1"),
        "editorLineNumber.foreground": readToken("--text-3"),
        "editorLineNumber.activeForeground": readToken("--accent"),
        "editor.lineHighlightBackground": "#22d3ee10",
        "editor.lineHighlightBorder": "#00000000",
        "editor.selectionBackground": "#22d3ee40",
        "editorCursor.foreground": readToken("--accent"),
        "editorGutter.background": "#00000000",
        "scrollbarSlider.background": "#82beff24",
        "scrollbarSlider.hoverBackground": "#82beff48",
        "editorWidget.background": "#04060c",
      },
    });
  };

  const applyDecorations = () => {
    const editor = editorRef.current;
    const monaco = monacoRef.current;
    const state = useProjectStore.getState();
    const doc = state.project?.document;
    const infos = state.project?.analysis.cells;
    if (!editor || !monaco || !doc || !infos) return;

    const roleByIndex = new Map(infos.map((info) => [info.index, info.role]));
    const decorations = doc.cells.flatMap((cell) => {
      const role = roleByIndex.get(cell.index) ?? "other";
      const header = {
        range: new monaco.Range(cell.headerLine, 1, cell.headerLine, 1),
        options: {
          isWholeLine: true,
          className: cell.kind === "code" ? "ns-cell-header" : "ns-cell-markdown",
          linesDecorationsClassName: `ns-role-bar ns-role-${role}`,
        },
      };
      if (cell.kind !== "code" || cell.lastLine < cell.firstLine) return [header];
      const body = {
        range: new monaco.Range(cell.firstLine, 1, cell.lastLine, 1),
        options: {
          isWholeLine: true,
          linesDecorationsClassName: `ns-role-bar ns-role-${role}`,
          className: cell.index === state.selectedCell ? "ns-cell-selected" : undefined,
        },
      };
      return [header, body];
    });
    decorationsRef.current?.clear();
    decorationsRef.current = editor.createDecorationsCollection(decorations);
  };

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    applyDecorations();

    const state = useProjectStore.getState();
    if (state.project && state.selectedCell !== null) {
      const line = editorLineFor(state.project.document, state.selectedCell);
      if (line) editor.revealLineNearTop(Math.max(1, line - 1));
    }

    editor.onDidChangeCursorPosition((event) => {
      // Only react to the user's own clicks and key presses, not to our reveals.
      if (event.source === "api" || event.source === "model") return;
      const current = useProjectStore.getState();
      if (!current.project) return;
      const cell = cellAtLine(current.project.document, event.position.lineNumber);
      if (cell && cell.index !== current.selectedCell) current.selectCell(cell.index, "editor");
    });
  };

  // Re-decorate when the notebook or the selection changes; scroll for map selections.
  useEffect(() => {
    applyDecorations();
    const state = useProjectStore.getState();
    const editor = editorRef.current;
    if (!editor || !state.project || selectedCell === null) return;
    if (state.selectionSource !== "map") return;
    const line = editorLineFor(state.project.document, selectedCell);
    if (line) editor.revealLineNearTop(Math.max(1, line - 1), 0);
  }, [document, cells, selectedCell]);

  if (!document) return null;

  return (
    <div className="min-h-0 flex-1">
      <Editor
        height="100%"
        language="python"
        theme={THEME}
        value={document.text}
        beforeMount={defineTheme}
        onMount={handleMount}
        loading={<p className="hud-label p-3">Loading editor…</p>}
        options={{
          readOnly: true,
          domReadOnly: true,
          minimap: { enabled: false },
          fontFamily: "var(--font-mono-face), monospace",
          fontSize: 12,
          lineHeight: 19,
          lineNumbersMinChars: 4,
          lineDecorationsWidth: 8,
          glyphMargin: false,
          folding: false,
          renderLineHighlight: "line",
          scrollBeyondLastLine: false,
          smoothScrolling: true,
          wordWrap: "off",
          contextmenu: false,
          overviewRulerLanes: 0,
          padding: { top: 6, bottom: 12 },
          scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
          guides: { indentation: false },
          stickyScroll: { enabled: false },
        }}
      />
    </div>
  );
}
