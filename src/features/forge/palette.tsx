"use client";

import { useState } from "react";
import { NeonButton, SegmentedControl } from "@/components/ui";
import { PALETTE, TEMPLATES, type ForgeFramework } from "@/core/forge";
import { FAMILY_COLOR_TOKEN } from "@/core/ops";
import { ProviderFields } from "@/features/ai/provider-fields";
import { currentModel, useAiStore } from "@/stores/ai-store";
import { useForgeStore } from "@/stores/forge-store";

const frameworks: ReadonlyArray<{ value: ForgeFramework; label: string }> = [
  { value: "pytorch", label: "PyTorch" },
  { value: "keras", label: "Keras" },
];

/** MIME type that marks a palette layer being dragged onto the canvas. */
export const FORGE_DRAG_TYPE = "application/x-neuroscope-op";

const fieldClass =
  "h-8 w-full min-w-0 rounded-control border border-line bg-glass-2 px-2 text-xs text-text-1 outline-none focus-visible:border-accent";

/** Forge's left panel: model name, framework, templates, the layers to add, and "describe it" with AI. */
export function Palette() {
  const name = useForgeStore((state) => state.doc.models[0]?.name ?? "");
  const framework = useForgeStore((state) =>
    state.doc.framework === "pytorch" ? "pytorch" : "keras",
  );
  const canUndo = useForgeStore((state) => state.past.length > 0);
  const canRedo = useForgeStore((state) => state.future.length > 0);
  const forge = useForgeStore.getState();

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto overflow-x-hidden p-3">
      <div className="flex items-center gap-2">
        <input
          // Remount when the name changes elsewhere (template, undo) so the field shows it.
          key={name}
          aria-label="Model name"
          defaultValue={name}
          className={fieldClass}
          onBlur={(event) => {
            if (event.target.value.trim() && event.target.value !== name)
              forge.rename(event.target.value);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") event.currentTarget.blur();
          }}
        />
        <NeonButton
          size="sm"
          disabled={!canUndo}
          onClick={forge.undo}
          aria-label="Undo"
          title="Undo (Ctrl+Z)"
        >
          ↶
        </NeonButton>
        <NeonButton
          size="sm"
          disabled={!canRedo}
          onClick={forge.redo}
          aria-label="Redo"
          title="Redo (Ctrl+Y)"
        >
          ↷
        </NeonButton>
      </div>

      <div className="flex items-center gap-2">
        <SegmentedControl
          label="Framework"
          options={frameworks}
          value={framework}
          onChange={forge.setFramework}
        />
        <select
          aria-label="Start from a template"
          className={fieldClass}
          value=""
          onChange={(event) => {
            if (event.target.value) forge.loadTemplate(event.target.value);
          }}
        >
          <option value="">Start from…</option>
          {TEMPLATES.map((template) => (
            <option key={template.id} value={template.id} title={template.description}>
              {template.label}
            </option>
          ))}
        </select>
      </div>

      <div>
        <p className="hud-label pb-1.5">Add a layer</p>
        <p className="pb-2 text-[11px] leading-snug text-text-3">
          Click, or drag onto the view. The layer goes after the selected one, or at the end.
        </p>
        <ul className="grid grid-cols-2 gap-1.5">
          {PALETTE.map((entry) => (
            <li key={entry.op}>
              <button
                type="button"
                draggable
                title={entry.hint}
                onClick={() => forge.add(entry.op)}
                onDragStart={(event) => {
                  event.dataTransfer.setData(FORGE_DRAG_TYPE, entry.op);
                  event.dataTransfer.effectAllowed = "copy";
                }}
                className="flex h-8 w-full cursor-grab items-center gap-2 rounded-control border border-line bg-glass-2 px-2 text-left text-xs text-text-1 transition-colors hover:border-accent/60 active:cursor-grabbing"
              >
                <span
                  aria-hidden
                  className="size-1.5 shrink-0 rounded-full"
                  style={{ background: `var(${FAMILY_COLOR_TOKEN[entry.family]})` }}
                />
                <span className="truncate">{entry.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <Describe />
    </div>
  );
}

/** Plain English → a model, through the AI provider chosen in the top bar. */
function Describe() {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const providers = useAiStore((state) => state.providers);
  const provider = useAiStore((state) => state.provider);
  const model = useAiStore((state) => currentModel(state));
  const consent = useAiStore((state) => state.consent);
  const localOnly = useAiStore((state) => state.localOnly);
  const run = useAiStore((state) => state.run);
  const { design, giveConsent, cancel } = useAiStore.getState();
  const info = providers?.find((entry) => entry.id === provider);
  const agreed = provider ? consent[provider] === true : false;
  const running = run.status === "running";

  return (
    <div className="border-t border-line pt-3">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="hud-label flex w-full items-center justify-between !text-text-1"
      >
        Describe a model <span aria-hidden>{open ? "▾" : "▸"}</span>
      </button>
      {open ? (
        <div className="flex flex-col gap-2 pt-2">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={3}
            placeholder="e.g. a CNN for 32×32 colour images with three conv blocks and 10 classes"
            className="w-full resize-y rounded-control border border-line bg-glass-2 p-2 text-xs leading-relaxed text-text-1 outline-none placeholder:text-text-3 focus-visible:border-accent"
          />
          <ProviderFields disabled={running} />
          {localOnly ? (
            <p className="text-[11px] leading-snug text-warn">
              Local only is on, so nothing can be sent.
            </p>
          ) : info ? (
            <label className="flex cursor-pointer items-start gap-2 text-[11px] leading-snug text-text-2">
              <input
                type="checkbox"
                className="mt-0.5 accent-[var(--accent)]"
                checked={agreed}
                onChange={(event) => giveConsent(info.id, event.target.checked)}
              />
              <span>Send this description to {info.label}. Only the text above is sent.</span>
            </label>
          ) : null}
          {run.log.length > 0 ? (
            <p
              aria-live="polite"
              className={`text-[11px] leading-snug ${run.status === "error" ? "text-error" : "text-text-3"}`}
            >
              {run.log[run.log.length - 1]!.text}
              {running && run.received > 0
                ? ` (${run.received.toLocaleString("en-US")} characters)`
                : ""}
            </p>
          ) : null}
          <div className="flex justify-end gap-2">
            {running ? (
              <NeonButton size="sm" variant="danger" onClick={cancel}>
                Cancel
              </NeonButton>
            ) : (
              <NeonButton
                size="sm"
                variant="primary"
                disabled={!text.trim() || !info || !model || !agreed || localOnly}
                onClick={() => void design(text.trim())}
              >
                Generate
              </NeonButton>
            )}
          </div>
          <p className="text-[11px] leading-snug text-text-3">
            The result replaces the model on screen (Undo brings the old one back). Every shape and
            count is recomputed by the engine.
          </p>
        </div>
      ) : null}
    </div>
  );
}
