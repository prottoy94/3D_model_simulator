"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Kbd, SegmentedControl } from "@/components/ui";

type Workspace = "scope" | "forge";

const workspaces: ReadonlyArray<{ value: Workspace; label: string }> = [
  { value: "scope", label: "Scope" },
  { value: "forge", label: "Forge" },
];

export function TopBar() {
  const pathname = usePathname();
  const router = useRouter();
  const current: Workspace = pathname.startsWith("/forge") ? "forge" : "scope";

  return (
    <header className="glass z-10 flex h-12 shrink-0 items-center gap-4 border-b border-line px-4">
      <Link href="/" className="flex items-center gap-2" aria-label="NeuroScope home">
        <span aria-hidden className="text-lg leading-none text-accent">
          ◈
        </span>
        <span className="font-display text-sm font-bold tracking-[0.22em] text-text-1">
          NEUROSCOPE
        </span>
      </Link>

      <SegmentedControl
        label="Workspace"
        options={workspaces}
        value={current}
        onChange={(next) => router.push(`/${next}`)}
      />

      <div className="min-w-0 flex-1 truncate text-center font-mono text-xs text-text-3">
        no notebook loaded
      </div>

      <span className="hidden items-center gap-1 text-xs text-text-3 md:flex">
        <Kbd>Ctrl</Kbd>
        <Kbd>K</Kbd>
      </span>

      <span className="flex items-center gap-2 text-xs text-text-2" title="AI engine (not connected yet)">
        <span aria-hidden className="size-2 rounded-full bg-text-3" />
        Fable 5.2 · offline
      </span>
    </header>
  );
}
