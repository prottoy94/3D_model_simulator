import { create } from "zustand";
import { extractSequentialModels, type Declined } from "@/core/fastpath";
import { buildRuntimeIR, type RuntimeIR } from "@/core/ir";
import {
  analyzeNotebook,
  NotebookParseError,
  parseNotebook,
  type Notebook,
  type NotebookAnalysis,
} from "@/core/notebook";
import { buildNotebookDocument, type NotebookDocument } from "@/core/notebook/document";
import { loadLatestProject, saveProject, sha256Hex } from "@/lib/project-db";
import { getPythonParser } from "@/lib/python-parser";

export type Project = {
  hash: string;
  name: string;
  notebook: Notebook;
  analysis: NotebookAnalysis;
  document: NotebookDocument;
  /** Models read from the code without AI (simple layer stacks only), shaped and verified. */
  ir: RuntimeIR | null;
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
};

async function buildProject(
  name: string,
  notebook: Notebook,
  originalBytes: number,
  hash: string,
  startedAt: number,
): Promise<Project> {
  const parser = await getPythonParser();
  const analysis = analyzeNotebook(notebook, parser);
  const fast = extractSequentialModels(notebook, analysis, parser);
  const built = fast.wire
    ? buildRuntimeIR(fast.wire, {
        summaries: analysis.cells.flatMap((cell) => cell.summaries),
        provenance: "heuristic",
      })
    : null;
  return {
    hash,
    name,
    notebook,
    analysis,
    document: buildNotebookDocument(notebook, analysis.cells),
    ir: built?.ok ? built.ir : null,
    declined: fast.declined,
    originalBytes,
    analysisMs: Math.round(performance.now() - startedAt),
  };
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
      const project = await buildProject(file.name, notebook, file.size, hash, startedAt);
      set({
        status: "ready",
        project,
        selectedCell: project.analysis.modelCells[0] ?? null,
        selectionSource: "map",
      });
      // Persistence is a convenience: a failure (private mode, full disk) must not block the UI.
      saveProject({
        hash,
        name: file.name,
        notebook,
        originalBytes: file.size,
        openedAt: Date.now(),
      }).catch(() => undefined);
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
    if (get().project || get().status === "loading") return;
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
      );
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
}));
