import type { Metadata } from "next";
import { ScopeShell } from "@/features/shell/scope-shell";
import { TopBar } from "@/features/shell/top-bar";

export const metadata: Metadata = { title: "Scope · NeuroScope" };

export default function ScopePage() {
  return (
    <div className="flex h-dvh flex-col">
      <TopBar />
      <ScopeShell />
    </div>
  );
}
