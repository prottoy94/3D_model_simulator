import { create } from "zustand";
import { exportCode } from "@/core/codegen/export";
import { generateListing, type CodeListing } from "@/core/codegen/listing";
import { buildRuntimeIR, type RuntimeIR, type WireIR } from "@/core/ir";
import type { ModelSummary } from "@/core/notebook";
import { EXAMPLE_IRS } from "@/core/ir/examples";
import { defaultCollapsed } from "@/core/layout";
import type { LensFocus } from "@/core/lens";
import { useUiStore } from "./ui-store";

export type ModelSource = "notebook" | "gallery" | "forge";
export type SelectionOrigin = "viewport" | "graph" | "editor" | "inspector";

type ModelState = {
  /** The architecture on screen. Every view reads this and nothing else (golden rule 1). */
  ir: RuntimeIR | null;
  source: ModelSource | null;
  /** Gallery models have no notebook, so their nodes link to this generated listing. */
  listing: CodeListing | null;
  activeModelId: string | null;
  /** A node id, or the id of a collapsed group. */
  selectedId: string | null;
  selectionOrigin: SelectionOrigin | null;
  hoveredIds: string[];
  collapsed: string[];
  showLabels: boolean;
  /** Lens: the part of the model that is isolated, or null for the whole model. */
  focus: LensFocus | null;
  /** What the code currently selected in the editor would focus (null focus = no layer there). */
  candidate: { focus: LensFocus | null; startLine: number; endLine: number } | null;
  /** Bumped to ask the 3D camera to frame the whole model. */
  fitRequest: number;

  /** The Wire IR behind `ir` and the summaries it was checked against, for re-verifying after an edit. */
  wire: WireIR | null;
  summaries: ModelSummary[];

  setIR: (
    ir: RuntimeIR | null,
    source: ModelSource,
    origin?: { wire: WireIR | null; summaries: ModelSummary[] },
    /** keepView: the same model after an edit, so the selection and camera stay. */
    options?: { keepView?: boolean },
  ) => void;
  /** Changes one symbol (an assumption) and recomputes every shape and count. */
  setSymbol: (name: string, value: number) => void;
  loadExample: (name: string) => boolean;
  setActiveModel: (modelId: string) => void;
  select: (id: string | null, origin: SelectionOrigin) => void;
  hover: (ids: string[]) => void;
  toggleGroup: (groupId: string) => void;
  expandAll: () => void;
  collapseAll: () => void;
  toggleLabels: () => void;
  setCandidate: (candidate: ModelState["candidate"]) => void;
  applyFocus: (focus: LensFocus) => void;
  clearFocus: () => void;
  requestFit: () => void;
};

/** The code shown next to a model that has no notebook: a listing for the gallery, a runnable script for Forge. */
function listingFor(
  ir: RuntimeIR,
  modelId: string,
  source: ModelSource | null,
): CodeListing | null {
  if (source === "gallery") return generateListing(ir, modelId);
  if (source === "forge") return exportCode(ir, modelId);
  return null;
}

const sameIds = (a: string[], b: string[]) =>
  a.length === b.length && a.every((id, i) => id === b[i]);

export const useModelStore = create<ModelState>((set, get) => ({
  ir: null,
  source: null,
  listing: null,
  activeModelId: null,
  selectedId: null,
  selectionOrigin: null,
  hoveredIds: [],
  collapsed: [],
  showLabels: true,
  fitRequest: 0,

  wire: null,
  summaries: [],

  setSymbol: (name, value) => {
    const { wire, summaries, activeModelId, collapsed, selectedId } = get();
    if (!wire || !Number.isFinite(value)) return;
    const next: WireIR = {
      ...wire,
      symbols: wire.symbols.map((symbol) =>
        symbol.name === name ? { ...symbol, value: JSON.stringify(value) } : symbol,
      ),
    };
    const built = buildRuntimeIR(next, { summaries });
    if (!built.ok) return;
    // Same model, same view: only the numbers change.
    set({
      ir: built.ir,
      wire: next,
      activeModelId: built.ir.models.some((model) => model.id === activeModelId)
        ? activeModelId
        : (built.ir.models[0]?.id ?? null),
      collapsed: collapsed.filter((id) => built.ir.groups[id]),
      selectedId:
        selectedId && (built.ir.nodes[selectedId] || built.ir.groups[selectedId])
          ? selectedId
          : null,
    });
  },

  setIR: (ir, source, origin, options) => {
    const previous = get();
    const keep = options?.keepView === true && previous.source === source && ir !== null;
    const activeModelId =
      (keep && ir.models.some((model) => model.id === previous.activeModelId)
        ? previous.activeModelId
        : null) ??
      ir?.models.find((model) => model.isPrimary)?.id ??
      ir?.models[0]?.id ??
      null;
    const stillThere = (id: string | null) =>
      id !== null && ir !== null && (ir.nodes[id] !== undefined || ir.groups[id] !== undefined);
    set({
      ir,
      wire: ir ? (origin?.wire ?? null) : null,
      summaries: origin?.summaries ?? [],
      source: ir ? source : null,
      activeModelId,
      listing: ir && activeModelId ? listingFor(ir, activeModelId, source) : null,
      selectedId: keep && stillThere(previous.selectedId) ? previous.selectedId : null,
      selectionOrigin: keep ? previous.selectionOrigin : null,
      hoveredIds: [],
      focus: null,
      candidate: null,
      collapsed: keep
        ? previous.collapsed.filter((id) => ir.groups[id] !== undefined)
        : ir
          ? defaultCollapsed(ir)
          : [],
      fitRequest: keep ? previous.fitRequest : previous.fitRequest + 1,
    });
  },

  loadExample: (name) => {
    const make = EXAMPLE_IRS[name];
    if (!make) return false;
    const built = buildRuntimeIR(make(), { provenance: "user" });
    if (!built.ok) return false;
    get().setIR(built.ir, "gallery");
    return true;
  },

  setActiveModel: (modelId) => {
    const { ir, source } = get();
    if (!ir || !ir.models.some((model) => model.id === modelId)) return;
    set({
      activeModelId: modelId,
      listing: listingFor(ir, modelId, source),
      selectedId: null,
      focus: null,
      candidate: null,
      hoveredIds: [],
      fitRequest: get().fitRequest + 1,
    });
  },

  select: (id, origin) => {
    // Picking something in a view brings its details to the front.
    // The Tensor tab stays: it follows the selection too.
    const ui = useUiStore.getState();
    if (id !== null && origin !== "inspector" && ui.inspectorTab === "notebook") {
      ui.setInspectorTab("layer");
    }
    set({ selectedId: id, selectionOrigin: origin });
  },

  hover: (ids) => {
    if (!sameIds(ids, get().hoveredIds)) set({ hoveredIds: ids });
  },

  toggleGroup: (groupId) => {
    const { collapsed, selectedId, ir } = get();
    const closing = !collapsed.includes(groupId);
    // A selection that disappears into (or out of) the group moves to what replaces it.
    let nextSelected = selectedId;
    if (selectedId && ir) {
      if (closing && ir.nodes[selectedId]) {
        for (
          let id: string | null = ir.nodes[selectedId]!.groupId;
          id;
          id = ir.groups[id]?.parentId ?? null
        ) {
          if (id === groupId) nextSelected = groupId;
        }
      } else if (!closing && selectedId === groupId) nextSelected = null;
    }
    set({
      collapsed: closing ? [...collapsed, groupId] : collapsed.filter((id) => id !== groupId),
      selectedId: nextSelected,
    });
  },

  // The model's size changes a lot, so the camera frames it again.
  expandAll: () => set({ collapsed: [], fitRequest: get().fitRequest + 1 }),

  collapseAll: () => {
    const { ir, activeModelId } = get();
    const root = ir?.models.find((model) => model.id === activeModelId)?.rootGroupId;
    if (!ir || !root) return;
    set({
      collapsed: ir.index.children[root]?.groups ?? [],
      selectedId: null,
      fitRequest: get().fitRequest + 1,
    });
  },

  toggleLabels: () => set({ showLabels: !get().showLabels }),

  focus: null,
  candidate: null,
  setCandidate: (candidate) => set({ candidate }),
  applyFocus: (focus) => {
    const { ir, collapsed } = get();
    if (!ir) return;
    // The focused layers must be on screen, so the groups around them are opened.
    const around = new Set<string>();
    for (const id of focus.nodeIds) {
      for (
        let group: string | null = ir.nodes[id]?.groupId ?? null;
        group;
        group = ir.groups[group]?.parentId ?? null
      ) {
        around.add(group);
      }
    }
    set({
      focus,
      collapsed: collapsed.filter((id) => !around.has(id)),
      selectedId: null,
      fitRequest: get().fitRequest + 1,
    });
  },
  clearFocus: () => {
    if (get().focus) set({ focus: null, fitRequest: get().fitRequest + 1 });
  },
  requestFit: () => set({ fitRequest: get().fitRequest + 1 }),
}));
