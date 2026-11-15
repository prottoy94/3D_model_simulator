import { create } from "zustand";
import { draftToWire, type Draft } from "@/core/ai/draft";
import { extractSequentialModels, type Declined } from "@/core/fastpath";
import { buildRuntimeIR, type RuntimeIR, type WireIR } from "@/core/ir";
import {
  analyzeNotebook,
  NotebookParseError,
  parseNotebook,
  type Notebook,
  type NotebookAnalysis,
} from "@/core/notebook";
import { buildNotebookDocument, type NotebookDocument } from "@/core/notebook/document";
import {
  loadLatestProject,
  loadProject,
  saveProject,
  sha256Hex,
  type AiSource,
} from "@/lib/project-db";
import { getPythonParser } from "@/lib/python-parser";
import { useModelStore } from "./model-store";

export type Project = {
  hash: string;
  name: string;
  notebook: Notebook;
  analysis: NotebookAnalysis;
  document: NotebookDocument;
  /** The architecture on screen: from the AI draft when there is one, else read without AI. */
  ir: RuntimeIR | null;
  /** The Wire IR `ir` was built from, kept so assumptions can be edited and re-verified. */
  wire: WireIR | null;
  /** What an AI provider returned for this notebook. Stored locally so it is never asked twice. */
  draft: Draft | null;
  aiSource: AiSource | null;
  /** Layer stacks the no-AI reader found but would not vouch for, and why. */
  declined: Declined[];
  originalBytes: number;
  /** How long parsing + analysis took, shown in the inspector. */
  analysisMs: number;
};

type ProjectState = {
  status: "idle" | "loading" | "ready" | "error";
  project: Project | null;
  error: string | null;
  /** Original index of the cell selected in the Notebook Map or the editor. */
  selectedCell: number | null;
  /** Where the last selection came from, so the editor does not scroll itself. */
  selectionSource: "map" | "editor" | null;
  openFile: (file: File) => Promise<boolean>;
  restoreLatest: () => Promise<void>;
  selectCell: (index: number | null, source: "map" | "editor") => void;
  /** Shows the open notebook's model again (after Forge used the views). */
  showCurrent: () => void;
  /** Shows an AI draft for the open notebook and saves it locally. */
  applyDraft: (draft: Draft, source: AiSource) => void;
  /** Drops the AI result and goes back to what can be read without AI. */
  discardDraft: () => void;
};

const summariesOf = (analysis: NotebookAnalysis) =>
  analysis.cells.flatMap((cell) => cell.summaries);

function buildFromWire(
  wire: WireIR | null,
  analysis: NotebookAnalysis,
  provenance: "ai" | "heuristic",
): RuntimeIR | null {
  if (!wire) return null;
  const built = buildRuntimeIR(wire, { summaries: summariesOf(analysis), provenance });
  return built.ok ? built.ir : null;
}

async function buildProject(
  name: string,
  notebook: Notebook,
  originalBytes: number,
  hash: string,
  startedAt: number,
  stored: { draft?: Draft; aiSource?: AiSource } = {},
): Promise<Project> {
  const parser = await getPythonParser();
  const analysis = analyzeNotebook(notebook, parser);
  const fast = extractSequentialModels(notebook, analysis, parser);

  // A saved AI result wins: it covers models the no-AI reader cannot read.
  let wire: WireIR | null = null;
  let ir: RuntimeIR | null = null;
  let draft: Draft | null = null;
  if (stored.draft) {
    wire = draftToWire(stored.draft);
    ir = buildFromWire(wire, analysis, "ai");
    if (ir) draft = stored.draft;
  }
  if (!ir) {
    wire = fast.wire;
    ir = buildFromWire(wire, analysis, "heuristic");
  }

  return {
    hash,
    name,
    notebook,
    analysis,
    document: buildNotebookDocument(notebook, analysis.cells),
    ir,
    wire: ir ? wire : null,
    draft,
    aiSource: draft ? (stored.aiSource ?? null) : null,
    declined: fast.declined,
    originalBytes,
    analysisMs: Math.round(performance.now() - startedAt),
  };
}

function show(project: Project): void {
  useModelStore.getState().setIR(project.ir, "notebook", {
    wire: project.wire,
    summaries: summariesOf(project.analysis),
  });
}

function persist(project: Project): void {
  // Persistence is a convenience: a failure (private mode, full disk) must not block the UI.
  saveProject({
    hash: project.hash,
    name: project.name,
    notebook: project.notebook,
    originalBytes: project.originalBytes,
    openedAt: Date.now(),
    draft: project.draft ?? undefined,
    aiSource: project.aiSource ?? undefined,
  }).catch(() => undefined);
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  status: "idle",
  project: null,
  error: null,
  selectedCell: null,
  selectionSource: null,

  openFile: async (file) => {
    if (!file.name.toLowerCase().endsWith(".ipynb")) {
      set({ status: "error", error: `"${file.name}" is not a .ipynb notebook.` });
      return false;
    }
    set({ status: "loading", error: null });
    try {
      const startedAt = performance.now();
      const notebook = parseNotebook(await file.text());
      if (notebook.stats.codeCells === 0) {
        throw new NotebookParseError("This notebook has no code cells to analyze.");
      }
      const hash = await sha256Hex(JSON.stringify(notebook.cells));
      // The same notebook opened before keeps its AI result: no second request.
      const earlier = await loadProject(hash).catch(() => undefined);
      const project = await buildProject(file.name, notebook, file.size, hash, startedAt, earlier);
      show(project);
      set({
        status: "ready",
        project,
        selectedCell: project.analysis.modelCells[0] ?? null,
        selectionSource: "map",
      });
      persist(project);
      return true;
    } catch (error) {
      const message =
        error instanceof NotebookParseError
          ? error.message
          : "The notebook could not be analyzed. Check that the file is not corrupted.";
      set({ status: "error", error: message });
      return false;
    }
  },

  restoreLatest: async () => {
    // A gallery model opened from the landing page must not be replaced by the last notebook.
    if (
      get().project ||
      get().status === "loading" ||
      useModelStore.getState().source === "gallery"
    )
      return;
    try {
      const stored = await loadLatestProject();
      if (!stored || get().project) return;
      set({ status: "loading" });
      const project = await buildProject(
        stored.name,
        stored.notebook,
        stored.originalBytes,
        stored.hash,
        performance.now(),
        stored,
      );
      show(project);
      set({
        status: "ready",
        project,
        selectedCell: project.analysis.modelCells[0] ?? null,
        selectionSource: "map",
      });
    } catch {
      set({ status: "idle" });
    }
  },

  selectCell: (index, source) => set({ selectedCell: index, selectionSource: source }),

  showCurrent: () => {
    const project = get().project;
    if (project) show(project);
    else useModelStore.getState().setIR(null, "notebook");
  },

  applyDraft: (draft, aiSource) => {
    const current = get().project;
    if (!current) return;
    const wire = draftToWire(draft);
    const ir = buildFromWire(wire, current.analysis, "ai");
    if (!ir) return;
    const project: Project = { ...current, ir, wire, draft, aiSource };
    show(project);
    set({ project });
    persist(project);
  },

  discardDraft: () => {
    const current = get().project;
    if (!current?.draft) return;
    void (async () => {
      const project = await buildProject(
        current.name,
        current.notebook,
        current.originalBytes,
        current.hash,
        performance.now(),
      );
      if (get().project?.hash !== current.hash) return;
      show(project);
      set({ project });
      persist(project);
    })();
  },
}));
