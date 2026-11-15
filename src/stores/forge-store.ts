import { create } from "zustand";
import { draftSchema, draftToWire, type Draft } from "@/core/ai/draft";
import {
  addNode,
  autoLabel,
  connect,
  disconnect,
  removeNode,
  renameModel,
  setFramework,
  setParam,
  TEMPLATES,
  type ForgeFramework,
} from "@/core/forge";
import { buildRuntimeIR } from "@/core/ir";
import type { OpKind } from "@/core/ops";
import { useModelStore } from "./model-store";

const STORAGE_KEY = "neuroscope.forge";
const HISTORY_LIMIT = 100;

type ForgeState = {
  doc: Draft;
  past: Draft[];
  future: Draft[];
  /** True once the saved document (if any) has been read from this browser. */
  ready: boolean;
  /** Loads the saved document and puts the model on screen. Call when Forge opens. */
  open: () => void;
  add: (op: OpKind) => void;
  remove: (id: string) => void;
  setParam: (id: string, key: string, value: unknown) => void;
  connect: (from: string, to: string) => void;
  disconnect: (from: string, to: string) => void;
  setFramework: (framework: ForgeFramework) => void;
  rename: (name: string) => void;
  loadTemplate: (id: string) => void;
  /** Replaces the document with a model designed elsewhere (the AI). */
  replace: (doc: Draft) => void;
  undo: () => void;
  redo: () => void;
};

// A new session starts with a small CNN in PyTorch (the default for research code).
const starter = () =>
  setFramework(TEMPLATES.find((entry) => entry.id === "small-cnn")!.build(), "pytorch");

function save(doc: Draft): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(doc));
  } catch {
    // Private mode: the model just does not survive a reload.
  }
}

function load(): Draft | null {
  try {
    const parsed = draftSchema.safeParse(
      JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null"),
    );
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}

/** Recomputes the model from the document and shows it. Selection and camera are kept unless asked. */
function show(doc: Draft, select?: string | null, fresh = false): void {
  const wire = draftToWire(doc);
  const built = buildRuntimeIR(wire, { provenance: "user" });
  if (!built.ok) return;
  const model = useModelStore.getState();
  model.setIR(built.ir, "forge", { wire, summaries: [] }, { keepView: !fresh });
  if (select !== undefined) model.select(select, "inspector");
}

export const useForgeStore = create<ForgeState>((set, get) => {
  /** Applies an edit: remembers the old document for undo, shows and saves the new one. */
  const commit = (next: Draft, select?: string | null, fresh = false) => {
    const { doc, past } = get();
    if (next === doc) return;
    set({ doc: next, past: [...past.slice(-HISTORY_LIMIT + 1), doc], future: [] });
    show(next, select, fresh);
    save(next);
  };

  return {
    doc: starter(),
    past: [],
    future: [],
    ready: false,

    open: () => {
      const doc = get().ready ? get().doc : (load() ?? get().doc);
      set({ doc, ready: true });
      show(doc, undefined, true);
    },

    add: (op) => {
      const selected = useModelStore.getState().selectedId;
      const { doc, id } = addNode(get().doc, op, selected);
      commit(doc, id);
    },
    remove: (id) => commit(removeNode(get().doc, id), null),
    setParam: (id, key, value) => commit(setParam(get().doc, id, key, value)),
    connect: (from, to) => commit(connect(get().doc, from, to)),
    disconnect: (from, to) => commit(disconnect(get().doc, from, to)),
    setFramework: (framework) => commit(setFramework(get().doc, framework), undefined, true),
    rename: (name) => commit(renameModel(get().doc, name.trim() || "MyModel")),
    loadTemplate: (id) => {
      const template = TEMPLATES.find((entry) => entry.id === id);
      if (!template) return;
      // A template keeps the framework the user is working in.
      commit(
        setFramework(template.build(), get().doc.framework === "pytorch" ? "pytorch" : "keras"),
        null,
        true,
      );
    },
    replace: (doc) => {
      // Forge edits one flat graph: keep the first model, drop groups and code locations.
      const model = doc.models[0]!;
      const designed: Draft = {
        ...doc,
        // The description says nothing about a framework, so the shapes are read as Keras-style.
        framework: "keras",
        models: [
          {
            ...model,
            groups: [],
            nodes: model.nodes.map((node) => {
              const params = { ...node.params };
              // Pooling to a vector is flat in both frameworks.
              if (node.op === "global_avg_pool" && params.keepDims === undefined)
                params.keepDims = false;
              return {
                ...node,
                params,
                group: null,
                lines: null,
                label: autoLabel(node.op, params),
              };
            }),
          },
        ],
      };
      // Then converted to the framework the user is working in.
      const framework = get().doc.framework === "pytorch" ? "pytorch" : "keras";
      commit(setFramework(designed, framework), null, true);
    },

    undo: () => {
      const { past, doc, future } = get();
      const previous = past[past.length - 1];
      if (!previous) return;
      set({ doc: previous, past: past.slice(0, -1), future: [doc, ...future] });
      show(previous);
      save(previous);
    },
    redo: () => {
      const { past, doc, future } = get();
      const next = future[0];
      if (!next) return;
      set({ doc: next, past: [...past, doc], future: future.slice(1) });
      show(next);
      save(next);
    },
  };
});
