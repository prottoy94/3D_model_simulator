"use client";

import { Chip, HudReadout, NeonButton } from "@/components/ui";
import { formatCompact, formatCount } from "@/core/format";
import { suggestFix } from "@/core/forge";
import type { RuntimeNode } from "@/core/ir";
import { FAMILY_COLOR_TOKEN, getOp } from "@/core/ops";
import { shapeText } from "@/core/shapes";
import { useForgeStore } from "@/stores/forge-store";
import { useModelStore } from "@/stores/model-store";

const ACTIVATIONS = ["linear", "relu", "gelu", "sigmoid", "tanh", "softmax"];
/** Settings that are not shown: internal, or not useful to edit by hand. */
const HIDDEN = new Set([
  "description",
  "layout",
  "dynamicDim",
  "momentum",
  "center",
  "scale",
  "selfAttention",
]);
/** Optional settings that have no default, so they would otherwise never appear. */
const OPTIONAL: Record<string, string[]> = { maxpool2d: ["stride"], avgpool2d: ["stride"] };
const LABELS: Record<string, string> = {
  filters: "Filters",
  kernel: "Kernel (h, w)",
  stride: "Stride (h, w)",
  padding: "Padding",
  dilation: "Dilation",
  groups: "Groups",
  bias: "Bias",
  activation: "Activation",
  pool: "Window (h, w)",
  units: "Units",
  rate: "Drop rate",
  shape: "Shape",
  axis: "Axis",
  numHeads: "Heads",
  keyDim: "Key size per head",
  valueDim: "Value size per head",
  dropout: "Attention dropout",
  vocab: "Vocabulary size",
  dim: "Vector size",
  epsilon: "Epsilon",
  keepDims: "Keep size-1 axes",
  startAxis: "Start axis",
  depthMultiplier: "Depth multiplier",
};

const fieldClass =
  "h-7 min-w-0 rounded-control border border-line bg-glass-2 px-2 text-right font-mono text-[12px] text-text-1 outline-none focus-visible:border-accent";

function Field({ node, name, value }: { node: RuntimeNode; name: string; value: unknown }) {
  const set = (next: unknown) => useForgeStore.getState().setParam(node.id, name, next);
  const number = (text: string) =>
    text.trim() === "" || !Number.isFinite(Number(text)) ? undefined : Number(text);
  const commitOnEnter = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") event.currentTarget.blur();
  };
  let control: React.ReactNode;

  if (typeof value === "boolean") {
    control = (
      <input
        type="checkbox"
        className="accent-[var(--accent)]"
        checked={value}
        onChange={(event) => set(event.target.checked)}
      />
    );
  } else if (name === "activation") {
    control = (
      <select
        className={`${fieldClass} w-28`}
        value={String(value)}
        onChange={(event) => set(event.target.value)}
      >
        {ACTIVATIONS.map((option) => (
          <option key={option}>{option}</option>
        ))}
      </select>
    );
  } else if (name === "padding") {
    const current = typeof value === "string" ? value : "explicit";
    control = (
      <select
        className={`${fieldClass} w-28`}
        value={current}
        onChange={(event) => set(event.target.value === "explicit" ? [1, 1] : event.target.value)}
      >
        <option value="same">same</option>
        <option value="valid">valid</option>
        <option value="explicit">
          {Array.isArray(value) ? `${value.join(", ")} px` : "1 px each side"}
        </option>
      </select>
    );
  } else if (name === "shape" || (Array.isArray(value) && value.length !== 2)) {
    const text = Array.isArray(value) ? value.join(", ") : "";
    control = (
      <input
        key={text}
        className={`${fieldClass} w-32`}
        defaultValue={text}
        aria-label={LABELS[name] ?? name}
        onKeyDown={commitOnEnter}
        onBlur={(event) => {
          const parts = event.target.value
            .split(/[\s,x×]+/)
            .filter(Boolean)
            .map(Number);
          if (
            parts.length > 0 &&
            parts.every((part) => Number.isInteger(part)) &&
            parts.join(", ") !== text
          )
            set(parts);
        }}
      />
    );
  } else if (Array.isArray(value) || value === undefined) {
    // A pair such as kernel or stride. An empty optional pair means "use the default".
    const pair = (Array.isArray(value) ? value : [undefined, undefined]) as Array<
      number | undefined
    >;
    control = (
      <span className="flex gap-1">
        {[0, 1].map((index) => (
          <input
            key={`${index}-${pair[index] ?? ""}`}
            type="number"
            min={1}
            className={`${fieldClass} w-14`}
            defaultValue={pair[index] ?? ""}
            placeholder="auto"
            aria-label={`${LABELS[name] ?? name} ${index === 0 ? "height" : "width"}`}
            onKeyDown={commitOnEnter}
            onBlur={(event) => {
              const next = number(event.target.value);
              if (next === pair[index]) return;
              if (next === undefined) return set(undefined);
              const other = pair[1 - index] ?? next;
              set(index === 0 ? [next, other] : [other, next]);
            }}
          />
        ))}
      </span>
    );
  } else if (typeof value === "number") {
    control = (
      <input
        key={value}
        type="number"
        step={Number.isInteger(value) ? 1 : "any"}
        className={`${fieldClass} w-24`}
        defaultValue={value}
        aria-label={LABELS[name] ?? name}
        onKeyDown={commitOnEnter}
        onBlur={(event) => {
          const next = number(event.target.value);
          if (next !== undefined && next !== value) set(next);
        }}
      />
    );
  } else {
    control = <span className="font-mono text-[12px] text-text-2">{String(value)}</span>;
  }

  return (
    <label className="flex items-center justify-between gap-3 py-1 text-xs text-text-3">
      <span className="min-w-0 truncate">{LABELS[name] ?? name}</span>
      {control}
    </label>
  );
}

/** Forge's Edit tab: the model's totals and problems, or the selected layer's settings. */
export function ForgeEditor() {
  const ir = useModelStore((state) => state.ir);
  const modelId = useModelStore((state) => state.activeModelId);
  const selectedId = useModelStore((state) => state.selectedId);
  if (!ir || !modelId) return null;

  const nodes = (ir.index.topoOrder[modelId] ?? []).map((id) => ir.nodes[id]!);
  const broken = Object.values(ir.nodes).filter((node) => node.status === "error");
  const totals = ir.verification.perModel[modelId]?.totals.computed;
  const node = selectedId ? ir.nodes[selectedId] : undefined;
  const select = (id: string) => useModelStore.getState().select(id, "inspector");

  if (!node) {
    return (
      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
        <Section title="Model">
          <HudReadout label="Layers" value={formatCount(nodes.length)} />
          <HudReadout
            label="Parameters"
            value={formatCount(totals?.total ?? 0)}
            source="computed"
          />
          <HudReadout
            label="FLOPs per sample"
            value={formatCompact(nodes.reduce((sum, item) => sum + item.flops, 0))}
            source="computed"
          />
          <HudReadout
            label="Problems"
            value={broken.length === 0 ? "none" : String(broken.length)}
            source={broken.length === 0 ? "summary" : "mismatch"}
          />
        </Section>
        {broken.length > 0 ? (
          <Section title="Fix these first">
            {broken.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => select(item.id)}
                className="block w-full py-1 text-left text-xs text-error hover:underline"
              >
                {item.label}: {item.issues.find((issue) => issue.severity === "error")?.message}
              </button>
            ))}
          </Section>
        ) : null}
        <p className="px-3 py-2 text-xs leading-relaxed text-text-3">
          Select a layer to change its settings. In the 2D view, drag from the dot under one card to
          another card to connect them, and double-click a connection to remove it.
        </p>
      </div>
    );
  }

  const spec = getOp(node.op);
  const fix = suggestFix(node);
  const keys = [
    ...Object.keys(node.params).filter((key) => !HIDDEN.has(key)),
    ...(OPTIONAL[node.op] ?? []).filter((key) => !(key in node.params)),
  ];
  const inputs = ir.index.inEdges[node.id] ?? [];
  const errors = node.issues.filter((issue) => issue.severity === "error");

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden [overflow-wrap:anywhere]">
      <Section title={spec?.title ?? node.op}>
        <div className="flex items-center justify-between gap-2 pb-1.5">
          <Chip color={`var(${FAMILY_COLOR_TOKEN[spec?.family ?? "custom"]})`}>{node.label}</Chip>
          <NeonButton
            size="sm"
            variant="danger"
            title="Remove this layer (Delete)"
            onClick={() => useForgeStore.getState().remove(node.id)}
          >
            Remove
          </NeonButton>
        </div>
        <HudReadout
          label="Output"
          value={
            node.outShape && node.status !== "error"
              ? shapeText(node.outShape)
              : "cannot be computed"
          }
          source={node.status === "error" ? "mismatch" : "computed"}
        />
        <HudReadout
          label="Parameters"
          value={formatCount(node.paramCount.trainable + node.paramCount.nonTrainable)}
        />
      </Section>

      {errors.length > 0 ? (
        <Section title="Problem">
          {errors.map((issue) => (
            <p key={issue.message} className="pb-1 text-xs leading-relaxed text-error">
              {issue.message}
            </p>
          ))}
          {fix ? <p className="text-xs leading-relaxed text-text-1">{fix}</p> : null}
        </Section>
      ) : null}

      {keys.length > 0 ? (
        <Section title="Settings">
          {keys.map((key) => (
            <Field key={`${node.id}-${key}`} node={node} name={key} value={node.params[key]} />
          ))}
        </Section>
      ) : null}

      {node.op !== "input" ? (
        <Section title="Reads from">
          {inputs.length === 0 ? (
            <p className="text-xs text-warn">Nothing is connected into this layer.</p>
          ) : null}
          {inputs.map((edge) => (
            <div
              key={edge.id}
              className="flex items-center justify-between gap-2 py-0.5 text-xs text-text-2"
            >
              <button
                type="button"
                className="truncate text-left hover:text-accent"
                onClick={() => select(edge.from)}
              >
                {ir.nodes[edge.from]?.label ?? edge.from}
              </button>
              <button
                type="button"
                aria-label={`Disconnect ${ir.nodes[edge.from]?.label ?? edge.from}`}
                title="Disconnect"
                className="text-text-3 hover:text-error"
                onClick={() => useForgeStore.getState().disconnect(edge.from, node.id)}
              >
                ×
              </button>
            </div>
          ))}
        </Section>
      ) : null}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-line px-3 py-2 last:border-b-0">
      <h3 className="hud-label pb-1">{title}</h3>
      {children}
    </section>
  );
}
