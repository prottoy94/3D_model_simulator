"use client";

import { useEffect, useState } from "react";
import { Group, Panel, Separator } from "react-resizable-panels";
import { GlassPanel, NeonButton, PanelHeader, SegmentedControl } from "@/components/ui";
import { exportNotebook, type ExportedCode } from "@/core/codegen/export";
import type { OpKind } from "@/core/ops";
import { CodeEditor } from "@/features/editor/code-editor";
import { LayerInspector } from "@/features/inspector/layer-inspector";
import { TensorInspector } from "@/features/inspector/tensor-inspector";
import { TimelineBar } from "@/features/shell/timeline-bar";
import { Viewport } from "@/features/viewport-3d";
import { cn } from "@/lib/cn";
import { useForgeStore } from "@/stores/forge-store";
import { useModelStore } from "@/stores/model-store";
import { ForgeEditor } from "./forge-editor";
import { FORGE_DRAG_TYPE, Palette } from "./palette";

type Tab = "edit" | "layer" | "tensor";
const tabs: ReadonlyArray<{ value: Tab; label: string }> = [
  { value: "edit", label: "Edit" },
  { value: "layer", label: "Layer" },
  { value: "tensor", label: "Tensor" },
];

/** Saves text as a file on the user's computer. */
function download(name: string, text: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

export function ForgeShell() {
  const open = useForgeStore((state) => state.open);
  const isForge = useModelStore((state) => state.source === "forge");
  const listing = useModelStore((state) => state.listing) as ExportedCode | null;
  const name = useForgeStore((state) => state.doc.models[0]?.name ?? "model");
  const [tab, setTab] = useState<Tab>("edit");
  const [dropping, setDropping] = useState(false);

  // Put Forge's model on screen when the page opens (Scope may have left another one there).
  useEffect(() => {
    open();
  }, [open]);

  // Delete removes the selected layer; Ctrl+Z / Ctrl+Y undo and redo. Typing in a field is left alone.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, select, textarea, [contenteditable], .monaco-editor")) return;
      const forge = useForgeStore.getState();
      const command = event.ctrlKey || event.metaKey;
      if (command && event.key.toLowerCase() === "z") {
        if (event.shiftKey) forge.redo();
        else forge.undo();
      } else if (command && event.key.toLowerCase() === "y") forge.redo();
      else if (event.key === "Delete" || event.key === "Backspace") {
        const selected = useModelStore.getState().selectedId;
        if (!selected) return;
        forge.remove(selected);
      } else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!isForge) return <main className="flex-1" />;
  const file = name.replace(/[^A-Za-z0-9_-]+/g, "_").toLowerCase() || "model";
  const suffix = listing?.framework === "pytorch" ? "torch" : "keras";

  return (
    <main className="relative flex min-h-0 flex-1 flex-col gap-2 p-2">
      <Group orientation="horizontal" className="min-h-0 flex-1">
        <Panel id="left" defaultSize="25%" minSize="16%" maxSize="45%" collapsible>
          <div className="flex h-full min-h-0 flex-col gap-2">
            <GlassPanel className="h-[55%]">
              <PanelHeader title="Forge" />
              <Palette />
            </GlassPanel>
            <GlassPanel className="flex-1">
              <PanelHeader
                title={`Code · ${listing?.framework === "pytorch" ? "PyTorch" : "Keras"}`}
                actions={
                  listing ? (
                    <>
                      <NeonButton
                        size="sm"
                        className="!h-6 !px-2"
                        title="Download the Python script"
                        onClick={() =>
                          download(`${file}_${suffix}.py`, listing.text, "text/x-python")
                        }
                      >
                        .py
                      </NeonButton>
                      <NeonButton
                        size="sm"
                        className="!h-6 !px-2"
                        title="Download as a Jupyter notebook"
                        onClick={() =>
                          download(
                            `${file}_${suffix}.ipynb`,
                            exportNotebook(listing, name),
                            "application/x-ipynb+json",
                          )
                        }
                      >
                        .ipynb
                      </NeonButton>
                      <NeonButton
                        size="sm"
                        className="!h-6 !px-2"
                        title="Download the model as NeuroScope IR (JSON)"
                        onClick={() =>
                          download(
                            `${file}.ir.json`,
                            JSON.stringify(useModelStore.getState().wire, null, 2),
                            "application/json",
                          )
                        }
                      >
                        IR
                      </NeonButton>
                    </>
                  ) : null
                }
              />
              <CodeEditor />
              {listing && listing.warnings.length > 0 ? (
                <p className="border-t border-line px-3 py-1.5 text-[11px] leading-snug text-warn">
                  {listing.warnings.length} layer{listing.warnings.length === 1 ? "" : "s"} could
                  not be written as code (marked TODO).
                </p>
              ) : null}
            </GlassPanel>
          </div>
        </Panel>

        <Handle />

        <Panel id="center" minSize="30%">
          <div
            className="relative h-full"
            onDragOver={(event) => {
              if (!event.dataTransfer.types.includes(FORGE_DRAG_TYPE)) return;
              event.preventDefault();
              event.dataTransfer.dropEffect = "copy";
              setDropping(true);
            }}
            onDragLeave={() => setDropping(false)}
            onDrop={(event) => {
              const op = event.dataTransfer.getData(FORGE_DRAG_TYPE);
              setDropping(false);
              if (!op) return;
              event.preventDefault();
              useForgeStore.getState().add(op as OpKind);
            }}
          >
            <Viewport />
            <div
              aria-hidden
              className={cn(
                "pointer-events-none absolute inset-0 rounded-panel border-2 border-dashed border-accent transition-opacity duration-150",
                dropping ? "opacity-100" : "opacity-0",
              )}
            />
          </div>
        </Panel>

        <Handle />

        <Panel id="right" defaultSize="22%" minSize="14%" maxSize="36%" collapsible>
          <GlassPanel className="h-full">
            <PanelHeader
              title="Inspector"
              actions={
                <SegmentedControl
                  label="Inspector tab"
                  options={tabs}
                  value={tab}
                  onChange={setTab}
                  className="!h-6 [&>button]:!px-2"
                />
              }
            />
            {tab === "edit" ? (
              <ForgeEditor />
            ) : tab === "layer" ? (
              <LayerInspector />
            ) : (
              <TensorInspector />
            )}
          </GlassPanel>
        </Panel>
      </Group>

      <TimelineBar />
    </main>
  );
}

function Handle() {
  return (
    <Separator className="group flex w-2 items-center justify-center outline-none">
      <span className="h-10 w-0.5 rounded-full bg-line transition-colors group-hover:bg-accent group-data-[separator=active]:bg-accent" />
    </Separator>
  );
}
