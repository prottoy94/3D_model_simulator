import type { Metadata } from "next";
import { ForgeShell } from "@/features/forge/forge-shell";
import { TopBar } from "@/features/shell/top-bar";

export const metadata: Metadata = { title: "Forge · NeuroScope" };

export default function ForgePage() {
  return (
    <div className="flex h-dvh flex-col">
      <TopBar />
      <ForgeShell />
    </div>
  );
}
