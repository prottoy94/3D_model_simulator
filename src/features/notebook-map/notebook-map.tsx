"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/cn";
import { useProjectStore } from "@/stores/project-store";
import { ROLE_META } from "./roles";

/**
 * The notebook at a glance: one row per cell, coloured by role. Model cells
 * glow; markdown cells act as section headings. Click a row to jump to its code.
 */
export function NotebookMap() {
  const cells = useProjectStore((state) => state.project?.analysis.cells);
  const selectedCell = useProjectStore((state) => state.selectedCell);
  const selectCell = useProjectStore((state) => state.selectCell);
  const selectedRef = useRef<HTMLButtonElement>(null);

  // Keep the selected row visible when the selection comes from the editor.
  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedCell]);

  if (!cells) return null;

  return (
    <ol className="min-h-0 flex-1 overflow-y-auto py-1" aria-label="Notebook cells">
      {cells.map((cell) => {
        const meta = ROLE_META[cell.role];
        const selected = cell.index === selectedCell;
        const isModel = cell.role === "model";

        if (cell.role === "markdown") {
          return (
            <li key={cell.index}>
              <button
                type="button"
                ref={selected ? selectedRef : undefined}
                onClick={() => selectCell(cell.index, "map")}
                className={cn(
                  "block w-full truncate px-3 pb-0.5 pt-2 text-left text-[11px] text-text-3 transition-colors hover:text-text-2",
                  selected && "text-text-1",
                )}
              >
                {cell.title || "(text)"}
              </button>
            </li>
          );
        }

        return (
          <li key={cell.index}>
            <button
              type="button"
              ref={selected ? selectedRef : undefined}
              onClick={() => selectCell(cell.index, "map")}
              aria-current={selected ? "true" : undefined}
              title={[meta.description, ...cell.signals].join("\n· ")}
              className={cn(
                "group relative flex w-full items-center gap-2 py-1 pl-3 pr-2 text-left transition-colors",
                selected ? "bg-accent/10" : "hover:bg-glass-2",
                !isModel && !selected && "opacity-70 hover:opacity-100",
              )}
            >
              <span
                aria-hidden
                className="absolute inset-y-0.5 left-0 w-[3px] rounded-r"
                style={{
                  background: meta.color,
                  boxShadow: isModel ? `0 0 10px ${meta.color}` : undefined,
                }}
              />
              <span className="w-7 shrink-0 font-mono text-[11px] tabular-nums text-text-3">
                {cell.index}
              </span>
              <span
                className={cn(
                  "w-12 shrink-0 font-display text-[10px] font-semibold uppercase tracking-[0.12em]",
                )}
                style={{ color: meta.color }}
              >
                {meta.label}
              </span>
              <span
                className={cn(
                  "min-w-0 flex-1 truncate font-mono text-[11px]",
                  isModel ? "font-semibold text-text-1" : "text-text-2",
                )}
              >
                {cell.title || "(empty)"}
              </span>
              {cell.summaries.length > 0 ? (
                <span
                  className="shrink-0 text-[10px] text-ok"
                  title="This cell printed a model summary"
                >
                  Σ
                </span>
              ) : null}
            </button>
          </li>
        );
      })}
    </ol>
  );
}
