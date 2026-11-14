"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { OpenNotebookButton } from "@/features/shell/open-notebook-button";
import { cn } from "@/lib/cn";
import { useProjectStore } from "@/stores/project-store";

/**
 * Notebook drop target on the landing page. The file is parsed and analyzed in
 * the browser, then the Scope workspace opens. Nothing is uploaded.
 */
export function DropZone() {
  const router = useRouter();
  const [dragging, setDragging] = useState(false);
  const openFile = useProjectStore((state) => state.openFile);
  const status = useProjectStore((state) => state.status);
  const error = useProjectStore((state) => state.error);

  const handleFile = async (file: File | undefined) => {
    if (file && (await openFile(file))) router.push("/scope");
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        void handleFile(event.dataTransfer.files[0]);
      }}
      className={cn(
        "glass relative w-full max-w-xl rounded-panel p-[1.5px] transition-shadow duration-200",
        dragging && "shadow-[0_0_60px_rgba(34,211,238,0.35)]",
      )}
      style={
        dragging
          ? {
              background:
                "linear-gradient(90deg, var(--accent), var(--accent-2), var(--accent)) 0 0 / 200% 100%",
              animation: "frame-charge 1.6s linear infinite",
            }
          : undefined
      }
    >
      <div className="flex flex-col items-center gap-4 rounded-[13px] border border-dashed border-line-strong bg-void/80 px-6 py-10 text-center">
        <span aria-hidden className="text-3xl text-accent">
          ◈
        </span>
        <div>
          <p className="font-display text-lg font-semibold text-text-1">
            Drop a notebook to see its mind
          </p>
          <p className="mt-1 text-sm text-text-2">
            Any <span className="font-mono text-text-1">.ipynb</span> with an ML, DL, CV or NLP
            model
          </p>
        </div>
        <OpenNotebookButton
          label="Choose a notebook"
          variant="primary"
          size="md"
          onOpened={() => router.push("/scope")}
        />
        <p aria-live="polite" className="min-h-5 text-xs">
          {status === "loading" ? <span className="text-text-2">Reading the notebook…</span> : null}
          {status === "error" && error ? <span className="text-error">{error}</span> : null}
          {status !== "loading" && status !== "error" ? (
            <span className="text-text-3">
              The file is read in your browser. Nothing is uploaded.
            </span>
          ) : null}
        </p>
      </div>
    </div>
  );
}
