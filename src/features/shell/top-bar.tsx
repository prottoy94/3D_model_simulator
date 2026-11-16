"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { SegmentedControl } from "@/components/ui";
import { AiMenu } from "@/features/ai/ai-menu";
import { CompareDialog } from "@/features/compare/compare-dialog";
import { useModelStore } from "@/stores/model-store";
import { useProjectStore } from "@/stores/project-store";
import { DisplayMenu } from "./display-menu";
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
  const notebookName = useProjectStore((state) => state.project?.name);
  const isGallery = useModelStore((state) => state.source === "gallery");
  const models = useModelStore((state) => state.ir?.models);
  const activeModelId = useModelStore((state) => state.activeModelId);
  const setActiveModel = useModelStore((state) => state.setActiveModel);
  const isForge = useModelStore((state) => state.source === "forge");
  const [comparing, setComparing] = useState(false);
  const projectName = isForge
    ? `Forge · ${models?.[0]?.name ?? "model"}`
    : isGallery
      ? `Gallery · ${models?.[0]?.name ?? "model"}`
      : notebookName;

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

      {models && models.length > 1 ? (
        <label className="flex items-center gap-2 text-xs text-text-3">
          Model
          <select
            value={activeModelId ?? ""}
            onChange={(event) => {
              setActiveModel(event.target.value);
              // Bring the code that defines the chosen model into view.
              const ir = useModelStore.getState().ir;
              const root = ir?.models.find((model) => model.id === event.target.value)?.rootGroupId;
              const cell = root ? ir?.groups[root]?.refs[0]?.cell : undefined;
              if (cell !== undefined) useProjectStore.getState().selectCell(cell, "map");
            }}
            className="h-8 rounded-control border border-line bg-glass-2 px-2 font-mono text-xs text-text-1 outline-none focus-visible:border-accent"
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      <OpenNotebookButton onOpened={() => router.push("/scope")} />

      <div data-tour="tools" className="flex items-center gap-1">
        {models && models.length > 0 ? (
          <button
            type="button"
            onClick={() => setComparing(true)}
            className="flex h-8 items-center rounded-control border border-transparent px-2 text-xs font-bold text-text-1 transition-colors hover:border-line"
          >
            Compare
          </button>
        ) : null}
        <DisplayMenu tour={current === "scope"} />
        <AiMenu />
      </div>

      {comparing ? <CompareDialog onClose={() => setComparing(false)} /> : null}
    </header>
  );
}
