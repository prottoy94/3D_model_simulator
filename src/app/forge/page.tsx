import type { Metadata } from "next";
import { GlassPanel } from "@/components/ui";
import { TopBar } from "@/features/shell/top-bar";

export const metadata: Metadata = { title: "Forge · NeuroScope" };

export default function ForgePage() {
  return (
    <div className="flex h-dvh flex-col">
      <TopBar />
      <main className="flex flex-1 items-center justify-center p-6">
        <GlassPanel className="max-w-md p-6 text-center">
          <h1 className="font-display text-xl font-semibold text-text-1">Forge</h1>
          <p className="mt-2 text-sm leading-relaxed text-text-2">
            Build a new model from code, a drag-and-drop graph, or plain English, and watch it run.
            The builder arrives in Phase 7 of the roadmap.
          </p>
        </GlassPanel>
      </main>
    </div>
  );
}
