"use client";

import { useEffect, useState } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";
import { GlassPanel, HudReadout, PanelHeader, SegmentedControl } from "@/components/ui";
import { CodeEditor } from "@/features/editor/code-editor";
import { LensChip } from "@/features/editor/lens-chip";
import { LayerInspector } from "@/features/inspector/layer-inspector";
import { NotebookInspector } from "@/features/inspector/notebook-inspector";
import { TensorInspector } from "@/features/inspector/tensor-inspector";
import { NotebookMap } from "@/features/notebook-map/notebook-map";
import { Viewport } from "@/features/viewport-3d";
import { cn } from "@/lib/cn";
import { useModelStore } from "@/stores/model-store";
import { usePrefsStore } from "@/stores/prefs-store";
import { useProjectStore } from "@/stores/project-store";
import { useUiStore, type InspectorTab } from "@/stores/ui-store";
import { OpenNotebookButton } from "./open-notebook-button";
import { TimelineBar } from "./timeline-bar";
import { Tour } from "./tour";

const modelTabs: ReadonlyArray<{ value: InspectorTab; label: string }> = [
  { value: "layer", label: "Layer" },
  { value: "tensor", label: "Tensor" },
];
const notebookTab = { value: "notebook", label: "Notebook" } as const;

export function ScopeShell() {
  const hasModel = useModelStore((state) => state.ir !== null);
  const isGallery = useModelStore((state) => state.source === "gallery");
  const inspectorTab = useUiStore((state) => state.inspectorTab);
  const setInspectorTab = useUiStore((state) => state.setInspectorTab);
  const hasProject = useProjectStore((state) => state.project !== null);
  const hasNotebook = hasProject && !isGallery;
  // A tab that does not apply to what is open falls back to Layer.
  const tab: InspectorTab =
    (inspectorTab === "notebook" && !hasNotebook) || (inspectorTab === "tensor" && !hasModel)
      ? "layer"
      : inspectorTab;
  const status = useProjectStore((state) => state.status);
  const error = useProjectStore((state) => state.error);
  const openFile = useProjectStore((state) => state.openFile);
  const restoreLatest = useProjectStore((state) => state.restoreLatest);
  const cellCount = useProjectStore((state) => state.project?.analysis.cells.length ?? 0);
  const [dragging, setDragging] = useState(false);

  // After a page reload, bring back the notebook that was open last.
  useEffect(() => {
    // Coming from Forge, its model is still on screen: show the notebook's again.
    if (useModelStore.getState().source === "forge") useProjectStore.getState().showCurrent();
    void restoreLatest();
  }, [restoreLatest]);

  // The first time a model is on screen, the tour introduces the workspace.
  const prefsLoaded = usePrefsStore((state) => state.loaded);
  const tourSeen = usePrefsStore((state) => state.tourSeen);
  useEffect(() => {
    if (prefsLoaded && !tourSeen && hasModel) usePrefsStore.getState().startTour();
  }, [prefsLoaded, tourSeen, hasModel]);

  return (
    <main
      className="relative flex min-h-0 flex-1 flex-col gap-2 p-2"
      onDragOver={(event) => {
        if (!event.dataTransfer.types.includes("Files")) return;
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const file = event.dataTransfer.files[0];
        if (file) void openFile(file);
      }}
    >
      <Group orientation="horizontal" className="min-h-0 flex-1">
        <Panel id="left" defaultSize="26%" minSize="16%" maxSize="50%" collapsible>
          <div className="flex h-full min-h-0 flex-col gap-2">
            <GlassPanel className="h-2/5">
              <PanelHeader
                title="Notebook map"
                actions={
                  hasProject ? (
                    <span className="font-mono text-[11px] text-text-3">{cellCount} cells</span>
                  ) : null
                }
              />
              {hasProject && !isGallery ? (
                <NotebookMap />
              ) : isGallery ? (
                <EmptyHint>
                  This is a gallery model, so there is no notebook. The code below is generated from
                  the model, one line per layer.
                </EmptyHint>
              ) : (
                <EmptyHint>Cells appear here, tagged by role. Model cells glow.</EmptyHint>
              )}
            </GlassPanel>
            <GlassPanel className="flex-1" data-tour="code">
              <PanelHeader title={isGallery ? "Code · generated" : "Code"} actions={<LensChip />} />
              {hasProject || isGallery ? (
                <CodeEditor />
              ) : (
                <EmptyHint>The notebook&apos;s code, linked to the Notebook Map.</EmptyHint>
              )}
            </GlassPanel>
          </div>
        </Panel>

        <RailHandle />

        <Panel id="center" minSize="30%">
          <Viewport />
        </Panel>

        <RailHandle />

        <Panel id="right" defaultSize="21%" minSize="14%" maxSize="36%" collapsible>
          <GlassPanel className="h-full" data-tour="inspector">
            <PanelHeader
              title="Inspector"
              actions={
                hasModel || hasNotebook ? (
                  <SegmentedControl
                    label="Inspector tab"
                    options={hasNotebook ? [...modelTabs, notebookTab] : modelTabs}
                    value={tab}
                    onChange={setInspectorTab}
                    className="!h-6 [&>button]:!px-2"
                  />
                ) : null
              }
            />
            {hasNotebook && tab === "notebook" ? (
              <NotebookInspector />
            ) : hasModel && tab === "tensor" ? (
              <TensorInspector />
            ) : hasModel || hasProject ? (
              <LayerInspector />
            ) : (
              <>
                <div className="px-3 py-2">
                  <HudReadout label="Notebook" value="–" source="placeholder" />
                  <HudReadout label="Model cells" value="–" source="placeholder" />
                </div>
                <EmptyHint>Open a notebook to see what it contains.</EmptyHint>
                <div className="px-3">
                  <OpenNotebookButton variant="primary" size="md" />
                </div>
              </>
            )}
            {status === "error" && error ? (
              <p role="alert" className="border-t border-line px-3 py-2 text-xs text-error">
                {error}
              </p>
            ) : null}
          </GlassPanel>
        </Panel>
      </Group>

      <div data-tour="timeline">
        <TimelineBar />
      </div>
      <Tour />

      <div
        aria-hidden={!dragging}
        className={cn(
          "pointer-events-none absolute inset-2 z-20 flex items-center justify-center rounded-panel border-2 border-dashed border-accent bg-void/80 transition-opacity duration-150",
          dragging ? "opacity-100" : "opacity-0",
        )}
      >
        <p className="font-display text-xl font-semibold uppercase tracking-[0.12em] text-accent">
          Drop the notebook to open it
        </p>
      </div>
    </main>
  );
}

function RailHandle() {
  return (
    <Separator className="group flex w-2 items-center justify-center outline-none">
      <span className="h-10 w-0.5 rounded-full bg-line transition-colors group-hover:bg-accent group-data-[separator=active]:bg-accent" />
    </Separator>
  );
}

function EmptyHint({ children }: { children: React.ReactNode }) {
  return <p className="px-3 py-3 text-xs leading-relaxed text-text-3">{children}</p>;
}
