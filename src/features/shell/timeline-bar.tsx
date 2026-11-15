"use client";

import { useEffect, useRef } from "react";
import { NeonButton } from "@/components/ui";
import { FAMILY_COLOR_TOKEN } from "@/core/ops";
import type { SimInput } from "@/core/sim/run";
import { cn } from "@/lib/cn";
import { useModelStore } from "@/stores/model-store";
import { simClock, useSimStore, type InputChoice } from "@/stores/sim-store";

const SPEEDS = [0.5, 1, 2, 4];
const selectClass =
  "h-8 rounded-control border border-line bg-glass-2 px-1.5 font-mono text-[11px] text-text-1 outline-none focus-visible:border-accent disabled:opacity-40";

/** Reads a picture file into RGB floats, at most 256 pixels a side. Nothing is uploaded. */
async function readImage(file: File): Promise<Extract<SimInput, { kind: "image" }>> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 256 / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("no canvas");
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const pixels = context.getImageData(0, 0, width, height).data;
  const rgb = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    for (let c = 0; c < 3; c++) rgb[i * 3 + c] = pixels[i * 4 + c]! / 255;
  }
  return { kind: "image", width, height, rgb };
}

/** Transport, scrubber and input picker for the forward-pass simulation. */
export function TimelineBar() {
  const plan = useSimStore((state) => state.plan);
  const status = useSimStore((state) => state.status);
  const playing = useSimStore((state) => state.playing);
  const waiting = useSimStore((state) => state.waiting);
  const step = useSimStore((state) => state.step);
  const ended = useSimStore((state) => state.ended);
  const computed = useSimStore((state) => state.computed);
  const total = useSimStore((state) => state.total);
  const speed = useSimStore((state) => state.speed);
  const loop = useSimStore((state) => state.loop);
  const follow = useSimStore((state) => state.follow);
  const input = useSimStore((state) => state.input);
  const imageName = useSimStore((state) => state.imageName);
  const estimate = useSimStore((state) => state.estimateSeconds);
  const error = useSimStore((state) => state.error);
  const anyIllustrative = useSimStore((state) =>
    Object.values(state.views).some((view) => view.illustrative),
  );
  const focused = useModelStore((state) => state.focus !== null);
  const actions = useSimStore.getState();

  const track = useRef<HTMLDivElement>(null);
  const fill = useRef<HTMLDivElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const file = useRef<HTMLInputElement>(null);

  // The playhead moves every frame, so it is positioned directly, not through React state.
  useEffect(() => {
    let frame = 0;
    const draw = () => {
      const fraction = plan && plan.total > 0 ? simClock.time / plan.total : 0;
      if (fill.current) fill.current.style.width = `${fraction * 100}%`;
      if (head.current) head.current.style.left = `${fraction * 100}%`;
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [plan]);

  // Space plays or pauses; the arrow keys step. Typing in a field or the code editor is left alone.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (!plan || event.ctrlKey || event.metaKey || event.altKey) return;
      if (target?.closest("input, select, textarea, button, [contenteditable], .monaco-editor"))
        return;
      const sim = useSimStore.getState();
      if (event.code === "Space") sim.toggle();
      else if (event.key === "ArrowRight") sim.stepBy(1);
      else if (event.key === "ArrowLeft") sim.stepBy(-1);
      else if (event.key === "Home") sim.seek(0);
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [plan]);

  const seekFromPointer = (clientX: number) => {
    const box = track.current?.getBoundingClientRect();
    if (!box || !plan) return;
    actions.seek(Math.max(0, Math.min(1, (clientX - box.left) / box.width)) * plan.total);
  };

  const current = plan?.steps[step];
  const disabled = !plan;
  let statusText: string;
  if (!plan) statusText = "No model to simulate";
  else if (status === "error") statusText = error ?? "The simulation failed";
  else if (status === "idle") {
    statusText =
      estimate > 8
        ? `Press play · computing this model takes about ${Math.round(estimate)} s`
        : "Press play to run a sample through the model";
  } else if (status === "computing") {
    statusText = `${waiting ? "Waiting for the next layer · " : ""}computing layer ${Math.min(computed + 1, total)} of ${total}`;
  } else {
    statusText = anyIllustrative
      ? "Real compute where possible · random-init weights · some layers illustrative"
      : "Real compute · random-init weights, not your trained model";
  }

  return (
    <footer className="glass flex min-h-14 shrink-0 flex-wrap items-center gap-x-3 gap-y-1.5 rounded-panel px-3 py-1.5">
      <div className="flex items-center gap-1">
        <NeonButton
          size="sm"
          disabled={disabled}
          aria-label="Back to the start"
          onClick={() => actions.seek(0)}
        >
          ⏮
        </NeonButton>
        <NeonButton
          size="sm"
          disabled={disabled}
          aria-label="Previous step"
          onClick={() => actions.stepBy(-1)}
        >
          ◀
        </NeonButton>
        <NeonButton
          size="sm"
          variant="primary"
          disabled={disabled}
          aria-label={playing ? "Pause" : "Play"}
          onClick={actions.toggle}
          className="w-10"
        >
          {playing ? "❚❚" : "▶"}
        </NeonButton>
        <NeonButton
          size="sm"
          disabled={disabled}
          aria-label="Next step"
          onClick={() => actions.stepBy(1)}
        >
          ▶
        </NeonButton>
      </div>

      <div className="flex min-w-48 flex-1 flex-col gap-1">
        <div
          ref={track}
          role="slider"
          tabIndex={disabled ? -1 : 0}
          aria-label="Position in the forward pass"
          aria-valuemin={1}
          aria-valuemax={plan?.steps.length ?? 1}
          aria-valuenow={step + 1}
          aria-valuetext={current ? `Step ${step + 1}: ${current.label}` : "No simulation"}
          className={cn("relative h-5 touch-none", disabled ? "opacity-40" : "cursor-pointer")}
          onPointerDown={(event) => {
            if (disabled) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            actions.pause();
            seekFromPointer(event.clientX);
          }}
          onPointerMove={(event) => {
            if (event.buttons === 1 && !disabled) seekFromPointer(event.clientX);
          }}
        >
          <div className="absolute inset-x-0 top-2 h-1 rounded-full bg-glass-2" />
          <div
            ref={fill}
            className="absolute left-0 top-2 h-1 rounded-full bg-accent/70"
            style={{ width: 0 }}
          />
          {plan?.steps.map((entry) => (
            <span
              key={entry.itemId}
              title={entry.label}
              className={cn(
                "absolute top-1 h-3 w-0.5 rounded-full",
                entry.index <= step ? "opacity-100" : "opacity-45",
              )}
              style={{
                left: `${(entry.start / plan.total) * 100}%`,
                background: `var(${FAMILY_COLOR_TOKEN[entry.family]})`,
              }}
            />
          ))}
          <div
            ref={head}
            className="absolute top-0 h-5 w-0.5 -translate-x-1/2 rounded-full bg-text-1 shadow-[0_0_8px_var(--accent)]"
            style={{ left: 0 }}
          />
        </div>
        <div className="flex items-baseline justify-between gap-3 text-[11px] leading-none">
          <span className="truncate font-mono text-text-1">
            {current
              ? `${focused ? "lens · " : ""}step ${ended ? plan!.steps.length : step + 1}/${plan!.steps.length} · ${current.label}`
              : "step –/–"}
          </span>
          <span
            className={cn("truncate text-right", status === "error" ? "text-error" : "text-text-3")}
          >
            {statusText}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-1.5">
        <label className="sr-only" htmlFor="sim-input">
          Sample input
        </label>
        <select
          id="sim-input"
          className={selectClass}
          disabled={disabled}
          value={input}
          title="What is fed into the model"
          onChange={(event) => {
            const choice = event.target.value as InputChoice;
            if (choice === "image" && !useSimStore.getState().image) file.current?.click();
            else actions.setInput(choice);
          }}
        >
          <option value="shapes">Input: shapes</option>
          <option value="stripes">Input: stripes</option>
          <option value="noise">Input: noise</option>
          <option value="image">
            {imageName ? `Input: ${imageName.slice(0, 14)}` : "Input: your image…"}
          </option>
        </select>
        <input
          ref={file}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => {
            const picked = event.target.files?.[0];
            event.target.value = "";
            if (picked)
              void readImage(picked).then((image) => actions.setImage(image, picked.name));
          }}
        />
        <NeonButton
          size="sm"
          disabled={disabled}
          onClick={() => file.current?.click()}
          title="Use one of your own pictures (it stays on this computer)"
        >
          Image…
        </NeonButton>
        <select
          className={selectClass}
          aria-label="Playback speed"
          disabled={disabled}
          value={speed}
          onChange={(event) => actions.setSpeed(Number(event.target.value))}
        >
          {SPEEDS.map((value) => (
            <option key={value} value={value}>
              {value}×
            </option>
          ))}
        </select>
        <NeonButton
          size="sm"
          disabled={disabled}
          aria-pressed={loop}
          variant={loop ? "primary" : "ghost"}
          onClick={() => actions.setLoop(!loop)}
        >
          Loop
        </NeonButton>
        <NeonButton
          size="sm"
          disabled={disabled}
          aria-pressed={follow}
          variant={follow ? "primary" : "ghost"}
          title="Keep the camera on the layer that is running"
          onClick={() => actions.setFollow(!follow)}
        >
          Follow
        </NeonButton>
      </div>
    </footer>
  );
}
