"use client";

import { Group, Panel, Separator } from "react-resizable-panels";
import { GlassPanel, HudReadout, PanelHeader } from "@/components/ui";
import { Viewport } from "@/features/viewport-3d";
import { TimelineBar } from "./timeline-bar";

export function ScopeShell() {
  return (
    <main className="flex min-h-0 flex-1 flex-col gap-2 p-2">
      <Group orientation="horizontal" className="min-h-0 flex-1">
        <Panel id="left" defaultSize="22%" minSize="14%" maxSize="40%" collapsible>
          <div className="flex h-full min-h-0 flex-col gap-2">
            <GlassPanel className="h-2/5">
              <PanelHeader title="Notebook map" />
              <EmptyHint>Cells appear here, tagged by role. Model cells glow.</EmptyHint>
            </GlassPanel>
            <GlassPanel className="flex-1">
              <PanelHeader title="Code" />
              <EmptyHint>The selected cell&apos;s code, linked to the 3D model.</EmptyHint>
            </GlassPanel>
          </div>
        </Panel>

        <RailHandle />

        <Panel id="center" minSize="30%">
          <Viewport />
        </Panel>

        <RailHandle />

        <Panel id="right" defaultSize="20%" minSize="14%" maxSize="36%" collapsible>
          <GlassPanel className="h-full">
            <PanelHeader title="Inspector" />
            <div className="px-3 py-2">
              <HudReadout label="Layer" value="–" source="placeholder" />
              <HudReadout label="Input shape" value="–" source="placeholder" />
              <HudReadout label="Output shape" value="–" source="placeholder" />
              <HudReadout label="Parameters" value="–" source="placeholder" />
              <HudReadout label="FLOPs" value="–" source="placeholder" />
            </div>
            <EmptyHint>Select a layer to see what it does to the tensor.</EmptyHint>
          </GlassPanel>
        </Panel>
      </Group>

      <TimelineBar />
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
