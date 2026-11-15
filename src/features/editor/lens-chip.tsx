"use client";

import { describeFocus } from "@/core/lens";
import { useModelStore } from "@/stores/model-store";

/**
 * Sits in the Code panel's header. With code selected it offers to isolate
 * and simulate the part of the model those lines define (shortcut: F).
 */
export function LensChip() {
  const candidate = useModelStore((state) => state.candidate);
  const focused = useModelStore((state) => state.focus !== null);
  const hasModel = useModelStore((state) => state.ir !== null);
  if (!hasModel) return null;
  if (!candidate) {
    return focused ? null : (
      <span className="text-[11px] text-text-3">Select code to focus on it</span>
    );
  }
  const lines =
    candidate.startLine === candidate.endLine
      ? `line ${candidate.startLine}`
      : `lines ${candidate.startLine}–${candidate.endLine}`;
  if (!candidate.focus) {
    return (
      <span
        className="truncate text-[11px] text-text-3"
        title="These lines do not create a layer of the model on screen."
      >
        No layer in {lines}
      </span>
    );
  }
  return (
    <button
      type="button"
      title={`${describeFocus(candidate.focus)} (press F)`}
      onClick={() => useModelStore.getState().applyFocus(candidate.focus!)}
      className="hud-label flex h-6 items-center gap-1.5 truncate rounded-control border border-accent/60 bg-accent/10 px-2 !text-[10px] !text-accent hover:bg-accent/20"
    >
      <span aria-hidden>◎</span> Simulate selection
    </button>
  );
}
