"use client";

import dynamic from "next/dynamic";
import { SegmentedControl } from "@/components/ui";
import { useProjectStore } from "@/stores/project-store";
import { useUiStore, type ViewMode } from "@/stores/ui-store";

// three.js needs a real WebGL context, so the scene never renders on the server.
const Scene = dynamic(() => import("./scene"), {
  ssr: false,
  loading: () => <ViewportMessage>Initializing viewport…</ViewportMessage>,
});

const viewModes: ReadonlyArray<{ value: ViewMode; label: string }> = [
  { value: "3d", label: "3D" },
  { value: "2d", label: "2D" },
];

export function Viewport() {
  const viewMode = useUiStore((state) => state.viewMode);
  const setViewMode = useUiStore((state) => state.setViewMode);
  const modelCells = useProjectStore((state) => state.project?.analysis.modelCells);

  // Honest status: the notebook is understood, but the model is not drawn before Phases 2–4.
  const banner = !modelCells
    ? "No model loaded · drop a notebook to begin"
    : modelCells.length === 0
      ? "No model cells found in this notebook"
      : `${modelCells.length} model ${modelCells.length === 1 ? "cell" : "cells"} found · 3D architecture not built yet`;

  return (
    <div className="relative h-full min-h-0 w-full overflow-hidden rounded-panel border border-line bg-deep">
      {viewMode === "3d" ? (
        <Scene />
      ) : (
        <ViewportMessage>The 2D graph view arrives in Phase 3.</ViewportMessage>
      )}

      <div className="pointer-events-none absolute inset-x-0 top-0 flex justify-center p-4">
        <p className="hud-label rounded-full border border-line bg-glass-1 px-3 py-1.5 text-center backdrop-blur">
          {banner}
        </p>
      </div>

      <div className="absolute bottom-3 left-3 flex items-center gap-2">
        <SegmentedControl
          label="View mode"
          options={viewModes}
          value={viewMode}
          onChange={setViewMode}
        />
      </div>
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
