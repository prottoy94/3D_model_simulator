"use client";

import { useEffect, useRef, useState } from "react";
import { GlassPanel } from "@/components/ui";
import { cn } from "@/lib/cn";
import { currentModel, useAiStore } from "@/stores/ai-store";
import { ProviderFields } from "./provider-fields";

/** The AI status in the top bar. Opens a small menu to pick the provider and model. */
export function AiMenu() {
  const providers = useAiStore((state) => state.providers);
  const provider = useAiStore((state) => state.provider);
  const model = useAiStore((state) => currentModel(state));
  const localOnly = useAiStore((state) => state.localOnly);
  const running = useAiStore((state) => state.run.status === "running");
  const loadError = useAiStore((state) => state.loadError);
  const loadProviders = useAiStore((state) => state.loadProviders);
  const setLocalOnly = useAiStore((state) => state.setLocalOnly);
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void loadProviders();
  }, [loadProviders]);

  // Close on a click outside or on Escape.
  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("pointerdown", onPointer);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const info = providers?.find((entry) => entry.id === provider);
  const ready = Boolean(info && model) && !localOnly;
  const label = localOnly
    ? "AI · local only"
    : providers === null
      ? "AI · loading"
      : info
        ? `AI · ${info.label.replace(/ \(.*\)$/, "")}`
        : "AI · no key";

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((value) => !value)}
        className="flex h-8 items-center gap-2 rounded-control border border-transparent px-2 text-xs font-bold text-text-1 transition-colors hover:border-line"
        title={ready ? `${info?.label} · ${model}` : "AI settings"}
      >
        <span
          aria-hidden
          className={cn(
            "size-2 rounded-full",
            running ? "animate-pulse bg-accent" : ready ? "bg-ok" : "bg-text-3",
          )}
        />
        {label}
        <span aria-hidden className="text-text-3">
          ▾
        </span>
      </button>

      {open ? (
        <GlassPanel
          raised
          role="dialog"
          aria-label="AI settings"
          className="absolute right-0 top-10 z-30 w-72 gap-3 p-3"
        >
          <p className="hud-label !text-text-1">AI engine</p>
          <ProviderFields disabled={running} />
          {info ? <p className="text-[11px] leading-snug text-text-3">{info.dataNote}</p> : null}
          {loadError ? <p className="text-[11px] leading-snug text-error">{loadError}</p> : null}
          <label className="flex cursor-pointer items-start gap-2 border-t border-line pt-3 text-xs text-text-2">
            <input
              type="checkbox"
              className="mt-0.5 accent-[var(--accent)]"
              checked={localOnly}
              disabled={running}
              onChange={(event) => setLocalOnly(event.target.checked)}
            />
            <span>
              <span className="font-semibold text-text-1">Local only</span>
              <br />
              Never send anything to an AI provider. Only simple layer stacks and gallery models are
              drawn.
            </span>
          </label>
        </GlassPanel>
      ) : null}
    </div>
  );
}
