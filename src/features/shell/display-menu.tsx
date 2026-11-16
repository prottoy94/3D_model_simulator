"use client";

import { useEffect, useRef, useState } from "react";
import { GlassPanel, SegmentedControl } from "@/components/ui";
import { usePrefsStore, type Palette, type Theme } from "@/stores/prefs-store";

const themes: ReadonlyArray<{ value: Theme; label: string }> = [
  { value: "dark", label: "Dark" },
  { value: "light", label: "Light" },
];
const palettes: ReadonlyArray<{ value: Palette; label: string }> = [
  { value: "default", label: "Standard" },
  { value: "safe", label: "Colorblind-safe" },
];

/** Reads the saved display choices once, on every page. Renders nothing. */
export function PrefsSync() {
  const load = usePrefsStore((state) => state.load);
  useEffect(() => load(), [load]);
  return null;
}

/** Theme, palette and the guided tour, in the top bar. */
export function DisplayMenu({ tour = true }: { tour?: boolean }) {
  const theme = usePrefsStore((state) => state.theme);
  const palette = usePrefsStore((state) => state.palette);
  const setTheme = usePrefsStore((state) => state.setTheme);
  const setPalette = usePrefsStore((state) => state.setPalette);
  const startTour = usePrefsStore((state) => state.startTour);
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

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

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((value) => !value)}
        className="flex h-8 items-center gap-1.5 rounded-control border border-transparent px-2 text-xs font-bold text-text-1 transition-colors hover:border-line"
      >
        Display
        <span aria-hidden className="text-text-3">
          ▾
        </span>
      </button>

      {open ? (
        <GlassPanel
          raised
          role="dialog"
          aria-label="Display settings"
          className="absolute right-0 top-10 z-30 w-72 gap-3 p-3"
        >
          <div className="flex flex-col gap-1.5">
            <p className="hud-label !text-text-1">Theme</p>
            <SegmentedControl label="Theme" options={themes} value={theme} onChange={setTheme} />
          </div>
          <div className="flex flex-col gap-1.5">
            <p className="hud-label !text-text-1">Layer colors</p>
            <SegmentedControl
              label="Layer colors"
              options={palettes}
              value={palette}
              onChange={setPalette}
            />
            <p className="text-[11px] leading-snug text-text-3">
              Colorblind-safe uses the Okabe–Ito colors. Layer status is always written out as text
              too, never shown by color alone.
            </p>
          </div>
          {tour ? (
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                startTour();
              }}
              className="border-t border-line pt-3 text-left text-xs font-semibold text-accent hover:underline"
            >
              Show the guided tour again
            </button>
          ) : null}
        </GlassPanel>
      ) : null}
    </div>
  );
}
