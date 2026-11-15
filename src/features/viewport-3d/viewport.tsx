"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef } from "react";
import { NeonButton, SegmentedControl } from "@/components/ui";
import { formatCount } from "@/core/format";
import { layoutModel3D } from "@/core/layout";
import { describeFocus } from "@/core/lens";
import { AnalyzePanel } from "@/features/ai/analyze-panel";
import { useAiStore } from "@/stores/ai-store";
import { useModelStore } from "@/stores/model-store";
import { useProjectStore } from "@/stores/project-store";
import { useUiStore, type ViewMode } from "@/stores/ui-store";
import { LabelOverlay, type LabelRegistry } from "./labels";

// three.js needs a real WebGL context, so the scene never renders on the server.
const Scene = dynamic(() => import("./scene"), {
  ssr: false,
  loading: () => <ViewportMessage>Initializing viewport…</ViewportMessage>,
});

// React Flow measures the DOM, so the 2D graph is client-only as well.
const GraphView = dynamic(() => import("@/features/graph-2d/graph-view"), {
  ssr: false,
  loading: () => <ViewportMessage>Loading graph…</ViewportMessage>,
});

const viewModes: ReadonlyArray<{ value: ViewMode; label: string }> = [
  { value: "3d", label: "3D" },
  { value: "2d", label: "2D" },
];

export function Viewport() {
  const viewMode = useUiStore((state) => state.viewMode);
  const setViewMode = useUiStore((state) => state.setViewMode);
  const modelCells = useProjectStore((state) => state.project?.analysis.modelCells);
  const ir = useModelStore((state) => state.ir);
  const source = useModelStore((state) => state.source);
  const activeModelId = useModelStore((state) => state.activeModelId);
  const collapsed = useModelStore((state) => state.collapsed);
  const showLabels = useModelStore((state) => state.showLabels);
  const { expandAll, collapseAll, toggleLabels, requestFit } = useModelStore.getState();

  const labels = useRef<LabelRegistry>(new Map());
  const focus = useModelStore((state) => state.focus);
  const clearFocus = useModelStore((state) => state.clearFocus);

  // Esc leaves a Lens focus, unless the key belongs to a field or the editor's own search.
  useEffect(() => {
    if (!focus) return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (event.key === "Escape" && !target?.closest("input, select, textarea")) clearFocus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focus, clearFocus]);
  const notebookHash = useProjectStore((state) => state.project?.hash);
  const panelChoice = useAiStore((state) => state.panelOpen);
  const analyzing = useAiStore((state) => state.run.status === "running");
  const setPanelOpen = useAiStore((state) => state.setPanelOpen);

  // A newly opened notebook goes back to the automatic rule below.
  useEffect(() => {
    setPanelOpen(null);
  }, [notebookHash, setPanelOpen]);
  const collapsedSet = useMemo(() => new Set(collapsed), [collapsed]);
  const layout = useMemo(
    () => (ir && activeModelId ? layoutModel3D(ir, activeModelId, collapsedSet) : null),
    [ir, activeModelId, collapsedSet],
  );

  const model = ir?.models.find((candidate) => candidate.id === activeModelId);
  const totals = model ? ir?.verification.perModel[model.id]?.totals : undefined;
  const hasGroups = (layout?.frames.length ?? 0) > 0 || collapsed.length > 0;

  // The Analyze panel opens by itself when a notebook has model code that could not be read without AI.
  const hasNotebook = notebookHash !== undefined && source === "notebook";
  const needsAi = hasNotebook && !ir && (modelCells?.length ?? 0) > 0;
  const showPanel = hasNotebook && (analyzing || (panelChoice ?? needsAi));

  // Honest status: say what is on screen and where its numbers come from.
  let banner: string;
  if (model && totals) {
    const trust =
      totals.match === true
        ? "matches the notebook's summary"
        : totals.match === false
          ? "differs from the notebook's summary"
          : source === "gallery"
            ? "gallery model"
            : source === "forge"
              ? "built in Forge"
              : "computed, no saved summary";
    banner = `${model.name} · ${formatCount(layout?.items.length ?? 0)} blocks · ${formatCount(totals.computed.total)} params · ${trust}`;
  } else if (!modelCells) {
    banner = "No model loaded · drop a notebook or pick one from the gallery";
  } else if (modelCells.length === 0) {
    banner = "No model cells found in this notebook";
  } else {
    banner = `${modelCells.length} model ${modelCells.length === 1 ? "cell" : "cells"} found · analyze with AI to build the architecture`;
  }

  return (
    <div className="relative h-full min-h-0 w-full overflow-hidden rounded-panel border border-line bg-deep">
      {viewMode === "3d" ? <Scene layout={layout} labels={labels} /> : <GraphView />}
      {viewMode === "3d" && layout ? <LabelOverlay layout={layout} registry={labels} /> : null}

      <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center p-4">
        {focus ? (
          <p className="hud-label pointer-events-auto flex items-center gap-3 rounded-full border border-accent/60 bg-glass-1 py-1 pl-3 pr-1 text-center !text-accent backdrop-blur">
            <span>Lens · {describeFocus(focus)}</span>
            <button
              type="button"
              onClick={clearFocus}
              title="Back to the whole model (Esc)"
              className="hud-label rounded-full border border-line px-2 py-1 !text-[10px] !text-text-1 hover:border-accent"
            >
              Exit
            </button>
          </p>
        ) : (
          <p className="hud-label rounded-full border border-line bg-glass-1 px-3 py-1.5 text-center backdrop-blur">
            {banner}
          </p>
        )}
      </div>

      <div className="absolute bottom-3 left-3 flex flex-wrap items-center gap-2">
        <SegmentedControl
          label="View mode"
          options={viewModes}
          value={viewMode}
          onChange={setViewMode}
        />
        {hasNotebook ? (
          <NeonButton
            size="sm"
            variant={needsAi ? "primary" : "ghost"}
            onClick={() => setPanelOpen(true)}
          >
            Analyze with AI
          </NeonButton>
        ) : null}
        {model ? (
          <>
            {hasGroups ? (
              <NeonButton size="sm" onClick={collapsed.length > 0 ? expandAll : collapseAll}>
                {collapsed.length > 0 ? "Expand all" : "Collapse groups"}
              </NeonButton>
            ) : null}
            {viewMode === "3d" ? (
              <>
                <NeonButton size="sm" aria-pressed={showLabels} onClick={toggleLabels}>
                  Labels {showLabels ? "on" : "off"}
                </NeonButton>
                <NeonButton size="sm" onClick={requestFit}>
                  Overview
                </NeonButton>
              </>
            ) : null}
          </>
        ) : null}
        {model ? (
          <p className="pointer-events-none basis-full text-[11px] leading-snug text-text-3">
            {viewMode === "3d"
              ? "Drag to move · right-drag to rotate · scroll to zoom · click a block to inspect it."
              : "Drag to move · scroll to zoom · click a card to inspect it."}{" "}
            Press play below to run a sample through the model.
          </p>
        ) : null}
      </div>

      {showPanel ? <AnalyzePanel onClose={() => setPanelOpen(false)} /> : null}
    </div>
  );
}

function ViewportMessage({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <p className="hud-label">{children}</p>
    </div>
  );
}
