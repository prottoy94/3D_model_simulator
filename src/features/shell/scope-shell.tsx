"use client";

import { useEffect, useState } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";
import { GlassPanel, HudReadout, PanelHeader } from "@/components/ui";
import { CodeEditor } from "@/features/editor/code-editor";
import { NotebookInspector } from "@/features/inspector/notebook-inspector";
import { NotebookMap } from "@/features/notebook-map/notebook-map";
import { Viewport } from "@/features/viewport-3d";
import { cn } from "@/lib/cn";
import { useProjectStore } from "@/stores/project-store";
import { OpenNotebookButton } from "./open-notebook-button";
import { TimelineBar } from "./timeline-bar";

export function ScopeShell() {
  const hasProject = useProjectStore((state) => state.project !== null);
  const status = useProjectStore((state) => state.status);
  const error = useProjectStore((state) => state.error);
  const openFile = useProjectStore((state) => state.openFile);
  const restoreLatest = useProjectStore((state) => state.restoreLatest);
  const cellCount = useProjectStore((state) => state.project?.analysis.cells.length ?? 0);
  const [dragging, setDragging] = useState(false);

  // After a page reload, bring back the notebook that was open last.
  useEffect(() => {
    void restoreLatest();
  }, [restoreLatest]);

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
              {hasProject ? (
                <NotebookMap />
              ) : (
                <EmptyHint>Cells appear here, tagged by role. Model cells glow.</EmptyHint>
              )}
            </GlassPanel>
            <GlassPanel className="flex-1">
              <PanelHeader title="Code" />
              {hasProject ? (
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
          <GlassPanel className="h-full">
            <PanelHeader title="Inspector" />
            {hasProject ? (
              <NotebookInspector />
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

      <TimelineBar />

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
