"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { SegmentedControl } from "@/components/ui";
import { useProjectStore } from "@/stores/project-store";
import { OpenNotebookButton } from "./open-notebook-button";

type Workspace = "scope" | "forge";

const workspaces: ReadonlyArray<{ value: Workspace; label: string }> = [
  { value: "scope", label: "Scope" },
  { value: "forge", label: "Forge" },
];

export function TopBar() {
  const pathname = usePathname();
  const router = useRouter();
  const current: Workspace = pathname.startsWith("/forge") ? "forge" : "scope";
  const projectName = useProjectStore((state) => state.project?.name);

  return (
    <header className="glass z-10 flex h-12 shrink-0 items-center gap-4 border-b border-line px-4">
      <Link href="/" className="flex items-center gap-2" aria-label="NeuroScope home">
        <span aria-hidden className="text-2xl leading-none text-accent">
          ◈
        </span>
        <span className="font-brand text-[13px] font-bold tracking-[0.28em] text-text-1">
          NEUROSCOPE
        </span>
      </Link>

      <SegmentedControl
        label="Workspace"
        options={workspaces}
        value={current}
        onChange={(next) => router.push(`/${next}`)}
      />

      <div
        className={`min-w-0 flex-1 truncate text-center font-mono text-xs ${
          projectName ? "text-text-1" : "text-text-3"
        }`}
        title={projectName}
      >
        {projectName ?? "no notebook loaded"}
      </div>

      <OpenNotebookButton onOpened={() => router.push("/scope")} />

      <span
        className="flex items-center gap-2 text-xs font-bold text-text-1"
        title="AI engine (not connected yet)"
      >
        <span aria-hidden className="size-2 rounded-full bg-text-3" />
        AI · not connected
      </span>
    </header>
  );
}
