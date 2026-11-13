"use client";

import { useRef, useState } from "react";
import { NeonButton } from "@/components/ui";
import { cn } from "@/lib/cn";

type DropState =
  | { kind: "idle" }
  | { kind: "accepted"; name: string; sizeMb: string }
  | { kind: "rejected"; reason: string };

/**
 * Notebook drop target. In Phase 0 it only validates the file type; Phase 1
 * wires it to the ingest pipeline. The file never leaves the browser here.
 */
export function DropZone() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [state, setState] = useState<DropState>({ kind: "idle" });

  const handleFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".ipynb")) {
      setState({ kind: "rejected", reason: `"${file.name}" is not a .ipynb notebook.` });
      return;
    }
    setState({
      kind: "accepted",
      name: file.name,
      sizeMb: (file.size / (1024 * 1024)).toFixed(1),
    });
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
        handleFile(event.dataTransfer.files[0]);
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
        <NeonButton variant="primary" onClick={() => inputRef.current?.click()}>
          Choose a notebook
        </NeonButton>
        <input
          ref={inputRef}
          type="file"
          accept=".ipynb"
          className="sr-only"
          aria-label="Choose a notebook file"
          onChange={(event) => handleFile(event.target.files?.[0])}
        />
        <p aria-live="polite" className="min-h-5 text-xs">
          {state.kind === "accepted" ? (
            <span className="text-ok">
              {state.name} ({state.sizeMb} MB) recognized. Notebook analysis is not built yet: it
              arrives in Phase 1.
            </span>
          ) : null}
          {state.kind === "rejected" ? <span className="text-error">{state.reason}</span> : null}
        </p>
      </div>
    </div>
  );
}
