"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { NeonButton } from "@/components/ui";
import { usePrefsStore } from "@/stores/prefs-store";

type Step = { target: string; title: string; text: string };

/** Each step points at the element marked `data-tour="<target>"`. */
const STEPS: ReadonlyArray<Step> = [
  {
    target: "viewport",
    title: "The model",
    text: "Every block is one layer, colored by its family. Data flows left to right. Drag to move, right-drag to rotate, scroll to zoom, and click a block to inspect it.",
  },
  {
    target: "code",
    title: "Code and model are linked",
    text: "Click a line to light up its layer, or click a layer to jump to its line. Select some code and press Lens to isolate just that part of the model.",
  },
  {
    target: "timeline",
    title: "Run a sample through it",
    text: "Press play to watch a forward pass, layer by layer. The values are really computed, but with random starting weights, not your trained ones.",
  },
  {
    target: "inspector",
    title: "The numbers",
    text: "Shapes, parameter counts and the receptive field of the selected layer. Each number says where it comes from: computed here, or checked against the notebook's saved summary.",
  },
  {
    target: "tools",
    title: "More tools",
    text: "Compare puts this model next to another one. Display switches to a light theme or colorblind-safe colors. Forge is where you build a model of your own.",
  },
];

type Box = { left: number; top: number; width: number; height: number };

const CARD_WIDTH = 320;
const MARGIN = 12;

/** A short guided tour: a ring around one part of the screen and a card that explains it. */
export function Tour() {
  const stepIndex = usePrefsStore((state) => state.tourStep);
  const setStep = usePrefsStore((state) => state.setTourStep);
  const end = usePrefsStore((state) => state.endTour);
  const [box, setBox] = useState<Box | null>(null);
  const next = useRef<HTMLButtonElement>(null);
  const step = stepIndex === null ? null : STEPS[stepIndex];

  useLayoutEffect(() => {
    if (!step) return;
    const measure = () => {
      const element = document.querySelector(`[data-tour="${step.target}"]`);
      const rect = element?.getBoundingClientRect();
      setBox(
        rect && rect.width > 0
          ? { left: rect.left, top: rect.top, width: rect.width, height: rect.height }
          : null,
      );
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [step]);

  useEffect(() => {
    if (!step) return;
    next.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      end();
    };
    // Capture phase: Escape closes the tour before it reaches Lens or a menu.
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [step, end]);

  if (!step || stepIndex === null) return null;

  const last = stepIndex === STEPS.length - 1;
  const viewport = { width: window.innerWidth, height: window.innerHeight };
  // The card goes beside the target when there is room, otherwise over its lower part.
  let left = (viewport.width - CARD_WIDTH) / 2;
  let top = viewport.height / 3;
  if (box) {
    const right = box.left + box.width;
    if (viewport.width - right >= CARD_WIDTH + 2 * MARGIN) left = right + MARGIN;
    else if (box.left >= CARD_WIDTH + 2 * MARGIN) left = box.left - CARD_WIDTH - MARGIN;
    else left = box.left + (box.width - CARD_WIDTH) / 2;
    // A tall target holds the card inside it; a short one has it above, or below near the top edge.
    if (box.height > 260) top = box.top + 48;
    else top = box.top > 220 ? box.top - 200 : box.top + box.height + MARGIN;
    left = Math.max(MARGIN, Math.min(viewport.width - CARD_WIDTH - MARGIN, left));
    top = Math.max(MARGIN, Math.min(viewport.height - 200, top));
  }

  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label="Guided tour">
      {/* The ring's huge shadow is what dims the rest of the screen. */}
      <div
        aria-hidden
        className="pointer-events-none absolute rounded-panel border-2 border-accent transition-all duration-200"
        style={{
          left: (box?.left ?? viewport.width / 2) - 3,
          top: (box?.top ?? viewport.height / 2) - 3,
          width: (box?.width ?? 0) + 6,
          height: (box?.height ?? 0) + 6,
          boxShadow: "0 0 0 200vmax color-mix(in srgb, var(--bg-void) 72%, transparent)",
        }}
      />
      <div
        className="glass absolute flex flex-col gap-2 rounded-panel border border-line-strong bg-glass-2 p-4"
        style={{ left, top, width: CARD_WIDTH }}
      >
        <p className="hud-label !text-accent">
          Tour · {stepIndex + 1} of {STEPS.length}
        </p>
        <h2 className="font-display text-base font-semibold text-text-1">{step.title}</h2>
        <p className="text-[13px] leading-relaxed text-text-2">{step.text}</p>
        <div className="flex items-center justify-between gap-2 pt-1">
          <button type="button" onClick={end} className="text-xs text-text-3 hover:text-text-1">
            Skip the tour
          </button>
          <div className="flex gap-2">
            {stepIndex > 0 ? (
              <NeonButton size="sm" onClick={() => setStep(stepIndex - 1)}>
                Back
              </NeonButton>
            ) : null}
            <NeonButton
              ref={next}
              size="sm"
              variant="primary"
              onClick={() => (last ? end() : setStep(stepIndex + 1))}
            >
              {last ? "Done" : "Next"}
            </NeonButton>
          </div>
        </div>
      </div>
    </div>
  );
}
