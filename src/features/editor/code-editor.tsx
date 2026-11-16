"use client";

import Editor, { type BeforeMount, type OnMount } from "@monaco-editor/react";
import { useEffect, useMemo, useRef } from "react";
import { cellAtLine, editorLineFor } from "@/core/notebook/document";
import { readToken } from "@/lib/tokens";
import { useModelStore } from "@/stores/model-store";
import { usePrefsStore, useSkin } from "@/stores/prefs-store";
import { useProjectStore } from "@/stores/project-store";
import { buildCodeLinks, idsAtLine, type CodeLink } from "./code-links";
import { focusForEditorLines } from "./lens-selection";

type MonacoEditor = Parameters<OnMount>[0];
type Monaco = Parameters<OnMount>[1];

const THEME = "neuroscope";

/**
 * The code behind the model. For a notebook this is the whole notebook as one
 * read-only Python document; for a gallery model it is a generated listing.
 *
 * It is linked both ways: selecting a cell or a layer scrolls here, and moving
 * the cursor or the mouse here selects or highlights the layer those lines define.
 */
export function CodeEditor() {
  const document = useProjectStore((state) => state.project?.document ?? null);
  const cells = useProjectStore((state) => state.project?.analysis.cells);
  const selectedCell = useProjectStore((state) => state.selectedCell);
  const ir = useModelStore((state) => state.ir);
  const listing = useModelStore((state) => state.listing);
  const activeModelId = useModelStore((state) => state.activeModelId);
  const selectedId = useModelStore((state) => state.selectedId);
  const hoveredIds = useModelStore((state) => state.hoveredIds);

  const links = useMemo(
    () => buildCodeLinks(ir, activeModelId, listing ? null : document, listing),
    [ir, activeModelId, document, listing],
  );
  const text = listing?.text ?? document?.text ?? "";

  const editorRef = useRef<MonacoEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const decorationsRef = useRef<ReturnType<MonacoEditor["createDecorationsCollection"]> | null>(
    null,
  );
  // Event handlers registered once at mount read the latest links from here.
  const linksRef = useRef<CodeLink[]>(links);
  const listingRef = useRef(listing !== null);
  useEffect(() => {
    linksRef.current = links;
    listingRef.current = listing !== null;
  }, [links, listing]);

  const defineTheme: BeforeMount = (monaco) => {
    const hex = (name: `--${string}`) => readToken(name).replace("#", "");
    monaco.editor.defineTheme(THEME, {
      base: usePrefsStore.getState().theme === "light" ? "vs" : "vs-dark",
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
        "editor.lineHighlightBackground": `${readToken("--accent")}14`,
        "editor.lineHighlightBorder": "#00000000",
        "editor.selectionBackground": `${readToken("--accent")}40`,
        "editorCursor.foreground": readToken("--accent"),
        "editorGutter.background": "#00000000",
        "scrollbarSlider.background": "#82beff24",
        "scrollbarSlider.hoverBackground": "#82beff48",
        "editorWidget.background": readToken("--bg-deep"),
      },
    });
  };

  // A new theme or palette: define the editor theme again from the tokens.
  const skin = useSkin();
  useEffect(() => {
    const monaco = monacoRef.current;
    if (!monaco) return;
    defineTheme(monaco);
    monaco.editor.setTheme(THEME);
  }, [skin]);

  const applyDecorations = () => {
    const editor = editorRef.current;
    const monaco = monacoRef.current;
    if (!editor || !monaco) return;
    const project = useProjectStore.getState();
    const model = useModelStore.getState();
    const doc = model.listing ? null : project.project?.document;
    const infos = project.project?.analysis.cells;

    const wholeLines = (start: number, end: number, className: string) => ({
      range: new monaco.Range(start, 1, end, 1),
      options: { isWholeLine: true, className },
    });

    const cellDecorations = (doc?.cells ?? []).flatMap((cell) => {
      const role = infos?.find((info) => info.index === cell.index)?.role ?? "other";
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
          className: cell.index === project.selectedCell ? "ns-cell-selected" : undefined,
        },
      };
      return [header, body];
    });

    const hovered = new Set(model.hoveredIds);
    const linkDecorations = linksRef.current.flatMap((link) => {
      // A group's range is only marked when the group itself is the selection.
      if (link.id === model.selectedId)
        return [wholeLines(link.start, link.end, "ns-ref-selected")];
      if (link.kind === "node" && hovered.has(link.id))
        return [wholeLines(link.start, link.end, "ns-ref-hovered")];
      return [];
    });

    decorationsRef.current?.clear();
    decorationsRef.current = editor.createDecorationsCollection([
      ...cellDecorations,
      ...linkDecorations,
    ]);
  };

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    applyDecorations();

    const state = useProjectStore.getState();
    if (!listingRef.current && state.project && state.selectedCell !== null) {
      const line = editorLineFor(state.project.document, state.selectedCell);
      if (line) editor.revealLineNearTop(Math.max(1, line - 1));
    }

    editor.onDidChangeCursorPosition((event) => {
      // Only react to the user's own clicks and key presses, not to our reveals.
      if (event.source === "api" || event.source === "model") return;
      const line = event.position.lineNumber;
      const current = useProjectStore.getState();
      if (!listingRef.current && current.project) {
        const cell = cellAtLine(current.project.document, line);
        if (cell && cell.index !== current.selectedCell) current.selectCell(cell.index, "editor");
      }
      const [id] = idsAtLine(linksRef.current, line);
      if (id && id !== useModelStore.getState().selectedId)
        useModelStore.getState().select(id, "editor");
    });

    // Lens: selected code becomes a candidate focus, offered in the panel header.
    editor.onDidChangeCursorSelection((event) => {
      const { selection } = event;
      const model = useModelStore.getState();
      if (selection.isEmpty()) {
        if (model.candidate) model.setCandidate(null);
        return;
      }
      // A selection that ends at the start of a line does not include that line.
      const endLine =
        selection.endColumn === 1 && selection.endLineNumber > selection.startLineNumber
          ? selection.endLineNumber - 1
          : selection.endLineNumber;
      model.setCandidate({
        focus: focusForEditorLines(selection.startLineNumber, endLine),
        startLine: selection.startLineNumber,
        endLine,
      });
    });
    editor.addAction({
      id: "neuroscope.lens",
      label: "Simulate selection",
      keybindings: [monaco.KeyCode.KeyF],
      run: () => {
        const model = useModelStore.getState();
        if (model.candidate?.focus) model.applyFocus(model.candidate.focus);
      },
    });

    editor.onMouseMove((event) => {
      const line = event.target.position?.lineNumber;
      useModelStore.getState().hover(line ? idsAtLine(linksRef.current, line) : []);
    });
    editor.onMouseLeave(() => useModelStore.getState().hover([]));
  };

  // Re-decorate when anything shown here changes; scroll for selections made in the map.
  useEffect(() => {
    applyDecorations();
    const state = useProjectStore.getState();
    const editor = editorRef.current;
    if (!editor || listing || !state.project || selectedCell === null) return;
    if (state.selectionSource !== "map") return;
    const line = editorLineFor(state.project.document, selectedCell);
    if (line) editor.revealLineNearTop(Math.max(1, line - 1), 0);
  }, [document, cells, selectedCell, listing]);

  useEffect(() => {
    applyDecorations();
  }, [links, hoveredIds]);

  // A layer picked in a view scrolls its code into the middle of the editor.
  useEffect(() => {
    applyDecorations();
    const editor = editorRef.current;
    if (!editor || !selectedId || useModelStore.getState().selectionOrigin === "editor") return;
    const first = linksRef.current.find((link) => link.id === selectedId);
    if (first) editor.revealLinesInCenterIfOutsideViewport(first.start, first.end, 0);
  }, [selectedId]);

  if (!text) return null;

  return (
    <div className="min-h-0 flex-1">
      <Editor
        height="100%"
        language="python"
        theme={THEME}
        value={text}
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
