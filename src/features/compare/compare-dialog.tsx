"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { NeonButton } from "@/components/ui";
import { modelStats, ratioText, type ModelStats } from "@/core/compare";
import { formatCompact, formatCount } from "@/core/format";
import { GALLERY } from "@/core/gallery";
import { buildRuntimeIR } from "@/core/ir";
import { FAMILY_COLOR_TOKEN } from "@/core/ops";
import { useModelStore } from "@/stores/model-store";

type Choice = { key: string; label: string; stats: () => ModelStats | null };

/** A gallery model's facts, built on demand (FireNet takes a moment). */
function galleryStats(name: string): ModelStats | null {
  const make = GALLERY[name];
  if (!make) return null;
  const built = buildRuntimeIR(make(), { provenance: "user" });
  const id = built.ok ? built.ir.models[0]?.id : undefined;
  return built.ok && id ? modelStats(built.ir, id) : null;
}

const GALLERY_LABELS: Record<string, string> = {
  alexnet: "AlexNet",
  vgg16: "VGG16",
  "mini-vit": "Mini Vision Transformer",
  firenet: "FireNet",
};

/** The model on screen next to another one: the notebook's other models or a gallery model. */
export function CompareDialog({ onClose }: { onClose: () => void }) {
  const ir = useModelStore((state) => state.ir);
  const activeModelId = useModelStore((state) => state.activeModelId);
  const closeButton = useRef<HTMLButtonElement>(null);

  const left = useMemo(
    () => (ir && activeModelId ? modelStats(ir, activeModelId) : null),
    [ir, activeModelId],
  );

  const choices = useMemo<Choice[]>(() => {
    const own: Choice[] = (ir?.models ?? [])
      .filter((model) => model.id !== activeModelId)
      .map((model) => ({
        key: `model:${model.id}`,
        label: `${model.name} · same notebook`,
        stats: () => modelStats(ir!, model.id),
      }));
    const gallery: Choice[] = Object.keys(GALLERY)
      .filter((name) => GALLERY_LABELS[name] !== left?.name)
      .map((name) => ({
        key: `gallery:${name}`,
        label: `${GALLERY_LABELS[name] ?? name} · gallery`,
        stats: () => galleryStats(name),
      }));
    return [...own, ...gallery];
  }, [ir, activeModelId, left?.name]);

  const [choiceKey, setChoiceKey] = useState<string | null>(null);
  const choice = choices.find((entry) => entry.key === choiceKey) ?? choices[0];
  const right = useMemo(() => choice?.stats() ?? null, [choice]);

  useEffect(() => {
    closeButton.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  if (!left) return null;

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-void/70 p-4 backdrop-blur-sm"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-label="Compare models"
        className="glass flex max-h-full w-full max-w-3xl flex-col overflow-hidden rounded-panel border border-line-strong bg-glass-2"
      >
        <header className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-line px-4">
          <h2 className="hud-label !font-bold !text-text-1">Compare models</h2>
          <NeonButton ref={closeButton} size="sm" onClick={onClose}>
            Close
          </NeonButton>
        </header>

        <div className="min-h-0 overflow-y-auto p-4">
          <div className="grid grid-cols-[9rem_1fr_1fr] items-end gap-x-4 gap-y-1 pb-3">
            <span />
            <p className="truncate font-display text-sm font-semibold text-accent">{left.name}</p>
            <label className="flex flex-col gap-1 text-[11px] text-text-3">
              Compare with
              <select
                value={choice?.key ?? ""}
                onChange={(event) => setChoiceKey(event.target.value)}
                className="h-8 rounded-control border border-line bg-glass-2 px-2 font-mono text-xs text-text-1 outline-none focus-visible:border-accent"
              >
                {choices.map((entry) => (
                  <option key={entry.key} value={entry.key}>
                    {entry.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {right ? (
            <>
              <Row label="Parameters" a={left.params} b={right.params} format={formatCount} />
              <Row label="Trainable" a={left.trainable} b={right.trainable} format={formatCount} />
              <Row label="FLOPs per sample" a={left.flops} b={right.flops} format={formatCompact} />
              <Row label="Layers" a={left.layers} b={right.layers} format={formatCount} />
              <Row label="Depth" a={left.depth} b={right.depth} format={formatCount} />
              <TextRow label="Input" a={left.inputs.join(", ")} b={right.inputs.join(", ")} />
              <TextRow label="Output" a={left.outputs.join(", ")} b={right.outputs.join(", ")} />
              <TextRow label="Heaviest layer" a={heaviest(left)} b={heaviest(right)} />
              <TextRow label="Framework" a={left.framework} b={right.framework} />

              <h3 className="hud-label pb-2 pt-5 !text-text-1">Where the parameters are</h3>
              <Shares stats={left} />
              <Shares stats={right} />

              <p className="pt-4 text-[11px] leading-relaxed text-text-3">
                Every number is computed from the layers. Depth is the number of layers on the
                longest path from input to output. FLOPs count one sample.
                {left.complete && right.complete
                  ? ""
                  : " A model with layers that could not be computed (a pretrained backbone, a custom layer) has totals that are a lower bound for FLOPs."}
              </p>
            </>
          ) : (
            <p className="text-xs text-text-3">That model could not be built.</p>
          )}
        </div>
      </section>
    </div>
  );
}

const heaviest = (stats: ModelStats) =>
  stats.heaviest
    ? `${stats.heaviest.label} · ${formatCompact(stats.heaviest.params)}`
    : "no weights";

function Row({
  label,
  a,
  b,
  format,
}: {
  label: string;
  a: number;
  b: number;
  format: (value: number) => string;
}) {
  const top = Math.max(a, b, 1);
  return (
    <div className="grid grid-cols-[9rem_1fr_1fr] items-center gap-x-4 border-t border-line py-2">
      <div>
        <p className="text-xs text-text-2">{label}</p>
        <p className="text-[10px] text-text-3">{ratioText(a, b)}</p>
      </div>
      <Bar value={a} top={top} text={format(a)} color="var(--accent)" />
      <Bar value={b} top={top} text={format(b)} color="var(--accent-2)" />
    </div>
  );
}

function Bar({
  value,
  top,
  text,
  color,
}: {
  value: number;
  top: number;
  text: string;
  color: string;
}) {
  return (
    <div className="min-w-0">
      <p className="truncate font-mono text-xs text-text-1">{text}</p>
      <div className="mt-1 h-1.5 rounded-full bg-line">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.max(1, (value / top) * 100)}%`, background: color }}
        />
      </div>
    </div>
  );
}

function TextRow({ label, a, b }: { label: string; a: string; b: string }) {
  return (
    <div className="grid grid-cols-[9rem_1fr_1fr] items-baseline gap-x-4 border-t border-line py-2">
      <p className="text-xs text-text-2">{label}</p>
      <p className="font-mono text-xs text-text-1 [overflow-wrap:anywhere]">{a}</p>
      <p className="font-mono text-xs text-text-1 [overflow-wrap:anywhere]">{b}</p>
    </div>
  );
}

/** One stacked bar: each family's share of the model's parameters. */
function Shares({ stats }: { stats: ModelStats }) {
  const withParams = stats.families.filter((share) => share.params > 0);
  return (
    <div className="pb-3">
      <p className="pb-1 text-xs text-text-2">{stats.name}</p>
      <div className="flex h-3 overflow-hidden rounded-full bg-line">
        {withParams.map((share) => (
          <div
            key={share.family}
            title={`${share.family}: ${formatCount(share.params)}`}
            style={{
              width: `${(share.params / Math.max(1, stats.params)) * 100}%`,
              background: `var(${FAMILY_COLOR_TOKEN[share.family]})`,
            }}
          />
        ))}
      </div>
      <ul className="flex flex-wrap gap-x-3 gap-y-0.5 pt-1.5">
        {withParams.map((share) => (
          <li key={share.family} className="flex items-center gap-1.5 text-[11px] text-text-3">
            <span
              aria-hidden
              className="size-2 rounded-sm"
              style={{ background: `var(${FAMILY_COLOR_TOKEN[share.family]})` }}
            />
            {share.family} {((share.params / Math.max(1, stats.params)) * 100).toFixed(1)}%
            <span className="text-text-3">({share.layers})</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
