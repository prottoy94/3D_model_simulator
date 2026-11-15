"use client";

import { useEffect, useRef } from "react";
import { Chip, HudReadout } from "@/components/ui";
import type { RuntimeNode } from "@/core/ir";
import { getOp } from "@/core/ops";
import { stateAt } from "@/core/sim/plan";
import type { Picture, TensorView } from "@/core/sim/view";
import { readToken } from "@/lib/tokens";
import { useModelStore } from "@/stores/model-store";
import { simClock, useSimStore } from "@/stores/sim-store";

const number = (value: number) =>
  Math.abs(value) >= 1000 || (Math.abs(value) < 0.001 && value !== 0)
    ? value.toExponential(2)
    : value.toFixed(3);

/** Draws a picture pixel for pixel; CSS scales it up without smoothing. */
function PictureCanvas({ picture, label }: { picture: Picture; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    canvas.width = picture.width;
    canvas.height = picture.height;
    context.putImageData(
      new ImageData(new Uint8ClampedArray(picture.rgba), picture.width, picture.height),
      0,
      0,
    );
  }, [picture]);
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={label}
      className="w-full rounded-[4px] border border-line [image-rendering:pixelated]"
      style={{ aspectRatio: `${picture.width} / ${picture.height}` }}
    />
  );
}

function Histogram({ view }: { view: TensorView }) {
  const peak = Math.max(1, ...view.histogram);
  return (
    <div>
      <div className="flex h-12 items-end gap-px" role="img" aria-label="Histogram of the values">
        {view.histogram.map((count, index) => (
          <span
            key={index}
            className="flex-1 rounded-t-[1px] bg-accent/70"
            style={{ height: `${Math.max(count > 0 ? 4 : 0, (count / peak) * 100)}%` }}
          />
        ))}
      </div>
      <div className="flex justify-between pt-0.5 font-mono text-[10px] text-text-3">
        <span>{number(view.stats.min)}</span>
        <span>{number(view.stats.max)}</span>
      </div>
    </div>
  );
}

/** A vector as bars. Probabilities list their largest entries by position. */
function Bars({ view }: { view: TensorView }) {
  const values = view.values!;
  if (view.probabilities) {
    const top = Array.from(values, (value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 6);
    return (
      <div className="flex flex-col gap-1">
        {top.map((entry) => (
          <div key={entry.index} className="flex items-center gap-2 text-[11px]">
            <span className="w-14 shrink-0 font-mono text-text-3">class {entry.index}</span>
            <span className="h-2 flex-1 overflow-hidden rounded-full bg-glass-2">
              <span
                className="block h-full rounded-full bg-op-act"
                style={{ width: `${entry.value * 100}%` }}
              />
            </span>
            <span className="w-12 shrink-0 text-right font-mono text-text-1">
              {(entry.value * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    );
  }
  const shown = Array.from(values.subarray(0, 96));
  const peak = Math.max(1e-9, ...shown.map(Math.abs));
  return (
    <div className="flex h-16 items-center gap-px" role="img" aria-label="The values as bars">
      {shown.map((value, index) => (
        <span key={index} className="relative h-full flex-1">
          <span
            className={`absolute inset-x-0 ${value >= 0 ? "bottom-1/2 bg-op-linear" : "top-1/2 bg-op-attn"}`}
            style={{ height: `${(Math.abs(value) / peak) * 50}%` }}
          />
        </span>
      ))}
    </div>
  );
}

type Sweep = {
  kernel: [number, number];
  stride: [number, number];
  input: [number, number];
  output: [number, number];
};

function sweepOf(node: RuntimeNode, view: TensorView): Sweep | null {
  const visualizer = getOp(node.op)?.visualizer;
  if (visualizer !== "conv-sweep" && visualizer !== "pool-window") return null;
  const inShape = node.inShapes[0];
  if (!inShape || view.shape.length !== 3 || !view.focus.input || !view.focus.output) return null;
  const pair = (value: unknown): [number, number] | null =>
    Array.isArray(value) && value.length === 2 ? [Number(value[0]), Number(value[1])] : null;
  const kernel = pair(node.params.kernel) ?? pair(node.params.pool);
  if (!kernel) return null;
  const dims = inShape.dims.slice(1).filter((dim): dim is number => typeof dim === "number");
  const [inH, inW] = inShape.layout === "NCHW" ? [dims[1]!, dims[2]!] : [dims[0]!, dims[1]!];
  return {
    kernel,
    stride: pair(node.params.stride) ?? (node.params.pool ? kernel : [1, 1]),
    input: [inH, inW],
    output: [view.shape[0]!, view.shape[1]!],
  };
}

/**
 * The sliding window of a convolution or pooling layer: the window moves over
 * the input (left) and the output fills in behind it (right). It follows the
 * simulation clock while this layer is the one running, and loops otherwise.
 */
function SweepCanvas({ node, view, sweep }: { node: RuntimeNode; view: TensorView; sweep: Sweep }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    const context = canvas?.getContext("2d");
    const input = view.focus.input;
    const output = view.focus.output;
    if (!canvas || !context || !input || !output) return;
    const size = 96;
    const gap = 16;
    canvas.width = size * 2 + gap;
    canvas.height = size;
    const toBitmap = (picture: Picture) => {
      const buffer = document.createElement("canvas");
      buffer.width = picture.width;
      buffer.height = picture.height;
      buffer
        .getContext("2d")
        ?.putImageData(
          new ImageData(new Uint8ClampedArray(picture.rgba), picture.width, picture.height),
          0,
          0,
        );
      return buffer;
    };
    const inputBitmap = toBitmap(input);
    const outputBitmap = toBitmap(output);
    const accent = readToken("--accent");
    const dim = readToken("--text-3");
    const [outH, outW] = sweep.output;
    const [inH, inW] = sweep.input;
    let frame = 0;

    const draw = (now: number) => {
      const sim = useSimStore.getState();
      const running =
        sim.plan?.steps[sim.step]?.memberIds.includes(node.id) &&
        sim.status !== "idle" &&
        !sim.ended;
      const state = sim.plan ? stateAt(sim.plan, simClock.time) : null;
      const progress =
        running && state
          ? state.phase === "travel"
            ? 0
            : state.phase === "compute"
              ? state.phaseProgress
              : 1
          : (now / 3500) % 1;
      const position = Math.min(outH * outW - 1, Math.floor(progress * outH * outW));
      const oy = Math.floor(position / outW);
      const ox = position % outW;

      context.imageSmoothingEnabled = false;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(inputBitmap, 0, 0, size, size);
      // The output so far: full rows above the window, and the current row up to it.
      const rowHeight = size / outH;
      context.save();
      context.beginPath();
      context.rect(size + gap, 0, size, oy * rowHeight);
      context.rect(size + gap, oy * rowHeight, ((ox + 1) / outW) * size, rowHeight);
      context.clip();
      context.drawImage(outputBitmap, size + gap, 0, size, size);
      context.restore();
      context.strokeStyle = dim;
      context.lineWidth = 1;
      context.strokeRect(size + gap + 0.5, 0.5, size - 1, size - 1);

      // The window on the input, and the single output value it produces.
      context.strokeStyle = accent;
      context.lineWidth = 1.5;
      const scaleY = size / inH;
      const scaleX = size / inW;
      const centreY = ((oy + 0.5) / outH) * inH;
      const centreX = ((ox + 0.5) / outW) * inW;
      context.strokeRect(
        (centreX - sweep.kernel[1] / 2) * scaleX,
        (centreY - sweep.kernel[0] / 2) * scaleY,
        Math.max(3, sweep.kernel[1] * scaleX),
        Math.max(3, sweep.kernel[0] * scaleY),
      );
      context.strokeRect(
        size + gap + (ox / outW) * size,
        oy * rowHeight,
        Math.max(2, size / outW),
        Math.max(2, rowHeight),
      );
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [node, view, sweep]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label="A window sliding over the input while the output fills in"
      className="w-full rounded-[4px] [image-rendering:pixelated]"
      style={{ aspectRatio: "208 / 96" }}
    />
  );
}

function KernelGrid({ kernel }: { kernel: NonNullable<TensorView["kernel"]> }) {
  const peak = Math.max(1e-9, ...Array.from(kernel.data, Math.abs));
  return (
    <div
      className="grid w-20 gap-px"
      style={{ gridTemplateColumns: `repeat(${kernel.width}, 1fr)` }}
      role="img"
      aria-label="The first filter's weights"
    >
      {Array.from(kernel.data, (weight, index) => (
        <span
          key={index}
          className="aspect-square rounded-[1px]"
          title={weight.toFixed(3)}
          style={{
            background: weight >= 0 ? "var(--op-conv)" : "var(--op-attn)",
            opacity: 0.15 + 0.85 * (Math.abs(weight) / peak),
          }}
        />
      ))}
    </div>
  );
}

/** The Tensor tab: what a layer actually produced for the sample input. */
export function TensorInspector() {
  const ir = useModelStore((state) => state.ir);
  const selectedId = useModelStore((state) => state.selectedId);
  const status = useSimStore((state) => state.status);
  const runningItem = useSimStore((state) => state.plan?.steps[state.step]);
  const views = useSimStore((state) => state.views);
  const estimate = useSimStore((state) => state.estimateSeconds);

  if (!ir) return <Hint>No model is on screen.</Hint>;
  if (status === "idle") {
    return (
      <Hint>
        Press play in the timeline at the bottom to run a sample image through the model. Each layer
        is really computed, with untrained (random) weights.
        {estimate > 8 ? ` This model takes about ${Math.round(estimate)} seconds to compute.` : ""}
      </Hint>
    );
  }

  // The selected layer, or the one the simulation is on. A collapsed group shows its last layer.
  let node: RuntimeNode | undefined;
  if (selectedId && ir.nodes[selectedId]) node = ir.nodes[selectedId];
  else if (selectedId && ir.groups[selectedId]) {
    const members = (ir.index.topoOrder[ir.groups[selectedId]!.modelId] ?? []).filter((id) => {
      for (
        let group: string | null = ir.nodes[id]!.groupId;
        group;
        group = ir.groups[group]?.parentId ?? null
      ) {
        if (group === selectedId) return true;
      }
      return false;
    });
    node = ir.nodes[members[members.length - 1] ?? ""];
  } else if (runningItem)
    node = ir.nodes[runningItem.memberIds[runningItem.memberIds.length - 1] ?? ""];
  if (!node) return <Hint>Select a layer to see what it produced.</Hint>;

  const view = views[node.id];
  if (!view) {
    return (
      <Hint>
        <span className="font-semibold text-text-1">{node.label}</span> has not been computed yet.
        Layers are computed in order while the simulation runs.
      </Hint>
    );
  }

  const sweep = sweepOf(node, view);
  const tilesNote =
    view.kind === "image" && view.tiles
      ? view.tiles.total === 1
        ? "The input picture."
        : `${view.tiles.shown} of ${view.tiles.total} channels, one small map each. Bright means a large value.`
      : view.kind === "tokens"
        ? `Each row is one of the ${view.shape[0]} tokens, each column one of its ${view.shape[1]} numbers.`
        : null;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden [overflow-wrap:anywhere]">
      <Section title={selectedId ? "Selected layer" : "Running layer"}>
        <p className="pb-1.5 text-[13px] font-semibold text-text-1">{node.label}</p>
        <div className="flex flex-wrap items-center gap-1.5 pb-1.5">
          {view.illustrative ? (
            <Chip color="var(--warn)">illustrative values</Chip>
          ) : (
            <Chip color="var(--ok)">really computed</Chip>
          )}
          <Chip>random-init weights</Chip>
        </div>
        <p className="text-xs leading-relaxed text-text-3">
          {view.note ??
            "Computed from the sample input with untrained weights, as Keras would initialize them. A trained model gives different values in the same shapes."}
        </p>
        <HudReadout label="Output shape" value={`(${view.shape.join(", ")})`} source="computed" />
        <HudReadout
          label="Computed in"
          value={view.ms < 1 ? "under 1 ms" : `${Math.round(view.ms)} ms`}
        />
      </Section>

      {sweep ? (
        <Section title="What this layer does">
          <SweepCanvas node={node} view={view} sweep={sweep} />
          <p className="pt-1 text-xs leading-relaxed text-text-2">
            A {sweep.kernel[0]}×{sweep.kernel[1]} window moves over the input (left) in steps of{" "}
            {sweep.stride[0]}. Each position gives one value of the output (right). One input
            channel and one output channel are shown.
          </p>
          {view.kernel ? (
            <div className="flex items-start gap-3 pt-2">
              <KernelGrid kernel={view.kernel} />
              <p className="text-[11px] leading-snug text-text-3">
                The first filter&apos;s weights for the first input channel. Cyan is positive,
                magenta negative. They are random until the model is trained.
              </p>
            </div>
          ) : null}
        </Section>
      ) : null}

      {view.attention ? (
        <Section title={`Attention · ${view.attention.heads} heads`}>
          <div className="grid grid-cols-2 gap-1.5">
            {view.attention.pictures.map((picture, head) => (
              <PictureCanvas key={head} picture={picture} label={`Attention of head ${head + 1}`} />
            ))}
          </div>
          <p className="pt-1 text-xs leading-relaxed text-text-2">
            One map per head. Row t shows how much token t looks at each of the{" "}
            {view.attention.tokens} tokens. With untrained weights the pattern is nearly even;
            training sharpens it.
          </p>
        </Section>
      ) : null}

      <Section title="Output">
        {view.picture ? <PictureCanvas picture={view.picture} label="The layer's output" /> : null}
        {view.values ? <Bars view={view} /> : null}
        {tilesNote ? (
          <p className="pt-1 text-[11px] leading-snug text-text-3">{tilesNote}</p>
        ) : null}
        {view.values && !view.probabilities ? (
          <p className="pt-1 text-[11px] leading-snug text-text-3">
            {view.values.length < (view.shape[0] ?? 0)
              ? `A sample of the ${view.shape[0]} values. `
              : ""}
            Amber bars are positive, magenta negative.
          </p>
        ) : null}
        {view.probabilities ? (
          <p className="pt-1 text-[11px] leading-snug text-text-3">
            The largest probabilities. An untrained model spreads them almost evenly.
          </p>
        ) : null}
      </Section>

      <Section title="Values">
        <Histogram view={view} />
        <HudReadout label="Mean" value={number(view.stats.mean)} />
        <HudReadout label="Spread (std)" value={number(view.stats.std)} />
        <HudReadout label="Exactly zero" value={`${(view.stats.zeroFraction * 100).toFixed(1)}%`} />
      </Section>
    </div>
  );
}

function Hint({ children }: { children: React.ReactNode }) {
  return <p className="px-3 py-3 text-xs leading-relaxed text-text-3">{children}</p>;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-line px-3 py-2 last:border-b-0">
      <h3 className="hud-label pb-1">{title}</h3>
      {children}
    </section>
  );
}
