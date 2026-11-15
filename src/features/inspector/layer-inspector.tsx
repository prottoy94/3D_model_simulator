"use client";

import { Chip, HudReadout, type ReadoutSource } from "@/components/ui";
import { formatCompact, formatCount } from "@/core/format";
import type { NodeStatus, RuntimeGroup, RuntimeIR, RuntimeNode } from "@/core/ir";
import { FAMILY_COLOR_TOKEN, getOp } from "@/core/ops";
import { shapeText } from "@/core/shapes";
import { useModelStore } from "@/stores/model-store";
import { useProjectStore } from "@/stores/project-store";

const STATUS: Record<NodeStatus, { label: string; color: string; text: string }> = {
  ok: { label: "verified", color: "var(--ok)", text: "Computed by the shape engine." },
  unverified: {
    label: "not verified",
    color: "var(--warn)",
    text: "Shown, but not confirmed by the shape engine.",
  },
  mismatch: {
    label: "mismatch",
    color: "var(--error)",
    text: "Computed, but the notebook's saved summary shows different numbers.",
  },
  error: {
    label: "error",
    color: "var(--error)",
    text: "The shape engine could not compute this layer.",
  },
};

function paramText(value: unknown): string {
  if (Array.isArray(value)) return value.length === 2 ? value.join("×") : `[${value.join(", ")}]`;
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number")
    return Number.isInteger(value) ? formatCount(value) : String(value);
  return value === undefined || value === "" ? "–" : String(value);
}

/** The Layer tab: the model overview, or the selected layer or group. */
export function LayerInspector() {
  const ir = useModelStore((state) => state.ir);
  const activeModelId = useModelStore((state) => state.activeModelId);
  const selectedId = useModelStore((state) => state.selectedId);
  const source = useModelStore((state) => state.source);

  if (!ir || !activeModelId) {
    return (
      <p className="px-3 py-3 text-xs leading-relaxed text-text-3">
        No architecture is on screen yet. If this notebook defines a model, use Analyze with AI in
        the viewport. Simple layer stacks and the gallery models on the home page need no AI.
      </p>
    );
  }

  const node = selectedId ? ir.nodes[selectedId] : undefined;
  const group = selectedId && !node ? ir.groups[selectedId] : undefined;
  return (
    <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden [overflow-wrap:anywhere]">
      {node ? (
        <NodeDetails ir={ir} node={node} />
      ) : group ? (
        <GroupDetails ir={ir} group={group} />
      ) : (
        <ModelOverview
          ir={ir}
          modelId={activeModelId}
          gallery={source === "gallery" || source === "forge"}
        />
      )}
    </div>
  );
}

function ModelOverview({
  ir,
  modelId,
  gallery,
}: {
  ir: RuntimeIR;
  modelId: string;
  gallery: boolean;
}) {
  const model = ir.models.find((candidate) => candidate.id === modelId);
  const report = ir.verification.perModel[modelId];
  if (!model || !report) return null;
  const { computed, match } = report.totals;
  const trust: ReadoutSource = match === null ? "computed" : match ? "summary" : "mismatch";
  const nodes = (ir.index.topoOrder[modelId] ?? []).map((id) => ir.nodes[id]!);
  const flops = nodes.reduce((sum, entry) => sum + entry.flops, 0);
  const flagged = nodes.filter((entry) => entry.status !== "ok");

  return (
    <>
      <Section title={model.name}>
        <p className="pb-1 text-xs leading-relaxed text-text-2">{ir.summary}</p>
        <HudReadout label="Layers" value={formatCount(nodes.length)} />
        <HudReadout label="Total params" value={formatCount(computed.total)} source={trust} />
        <HudReadout label="Trainable" value={formatCount(computed.trainable)} source={trust} />
        <HudReadout
          label="Non-trainable"
          value={formatCount(computed.nonTrainable)}
          source={trust}
        />
        <HudReadout label="FLOPs per sample" value={formatCompact(flops)} source="computed" />
        <p className="pt-1 text-xs leading-relaxed text-text-3">
          {match === true
            ? `Every number is computed from the layers and equals the notebook's saved summary (${report.rows.length} rows compared).`
            : match === false
              ? "The computed totals differ from the notebook's saved summary. Flagged layers are listed below."
              : gallery
                ? "Every number is computed from the layers."
                : "Computed from the layers. The notebook has no saved summary to compare with."}
        </p>
      </Section>

      {flagged.length > 0 ? (
        <Section title={`Needs attention · ${flagged.length}`}>
          {flagged.slice(0, 12).map((entry) => (
            <button
              key={entry.id}
              type="button"
              onClick={() => useModelStore.getState().select(entry.id, "inspector")}
              className="flex w-full items-baseline justify-between gap-2 py-0.5 text-left text-xs text-text-2 hover:text-text-1"
            >
              <span className="truncate">{entry.label}</span>
              <span style={{ color: STATUS[entry.status].color }}>
                {STATUS[entry.status].label}
              </span>
            </button>
          ))}
        </Section>
      ) : null}

      <AiSourceNote />

      {ir.assumptions.length > 0 ? (
        <Section title="Assumptions">
          <p className="pb-1 text-[11px] leading-snug text-text-3">
            Values the code does not state. Change one and every shape and count is recomputed.
          </p>
          {ir.assumptions.map((entry) => (
            <Assumption
              key={entry.symbol}
              name={entry.symbol}
              value={entry.value}
              note={entry.source}
            />
          ))}
        </Section>
      ) : null}

      {[...ir.warnings, ...ir.issues.map((issue) => issue.message)].length > 0 ? (
        <Section title="Notes">
          {[...ir.warnings, ...ir.issues.map((issue) => issue.message)].map((message) => (
            <p key={message} className="pb-1 text-xs leading-relaxed text-warn">
              {message}
            </p>
          ))}
        </Section>
      ) : null}

      <p className="px-3 py-2 text-xs leading-relaxed text-text-3">
        Click a block in the view, or a line of code, to inspect that layer.
      </p>
    </>
  );
}

/** Says which AI described this model, and lets the user drop that result. */
function AiSourceNote() {
  const aiSource = useProjectStore((state) => state.project?.aiSource ?? null);
  const hasDraft = useProjectStore((state) => state.project?.draft != null);
  if (!hasDraft) return null;
  return (
    <Section title="Source">
      <p className="text-xs leading-relaxed text-text-2">
        The layers were described by{" "}
        {aiSource ? `${aiSource.provider} (${aiSource.model})` : "an AI model"} from the
        notebook&apos;s code. Every shape and count shown is recomputed by the engine, not taken
        from the AI.
      </p>
      <button
        type="button"
        onClick={() => useProjectStore.getState().discardDraft()}
        className="mt-1.5 text-xs text-text-3 hover:text-error"
      >
        Remove this AI result
      </button>
    </Section>
  );
}

function Assumption({ name, value, note }: { name: string; value: string; note: string }) {
  const editable = useModelStore((state) => state.wire !== null) && /^-?\d+(\.\d+)?$/.test(value);
  const commit = (text: string) => {
    const next = Number(text);
    if (text.trim() !== "" && Number.isFinite(next) && String(next) !== value) {
      useModelStore.getState().setSymbol(name, next);
    }
  };
  return (
    <div className="pb-1.5">
      <div className="flex items-center justify-between gap-3 py-0.5">
        <span className="min-w-0 truncate font-mono text-xs text-text-3">{name}</span>
        {editable ? (
          <input
            // Remount when the stored value changes so the field shows it.
            key={value}
            type="number"
            defaultValue={value}
            aria-label={name}
            onBlur={(event) => commit(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") event.currentTarget.blur();
            }}
            className="h-7 w-24 rounded-control border border-line bg-glass-2 px-2 text-right font-mono text-[13px] text-text-1 outline-none focus-visible:border-accent"
          />
        ) : (
          <span className="font-mono text-[13px] text-text-1">{value}</span>
        )}
      </div>
      <p className="text-[11px] leading-snug text-text-3">{note}</p>
    </div>
  );
}

function NodeDetails({ ir, node }: { ir: RuntimeIR; node: RuntimeNode }) {
  const spec = getOp(node.op);
  const status = STATUS[node.status];
  const trust: ReadoutSource =
    node.status === "mismatch" || node.status === "error"
      ? "mismatch"
      : node.summaryRow && node.status === "ok"
        ? "summary"
        : "computed";
  const total = node.paramCount.trainable + node.paramCount.nonTrainable;
  const row = node.summaryRow;

  let explanation = node.explanation;
  if (!explanation && spec && node.outShape && node.status !== "error") {
    try {
      explanation = spec.explain(node.params, { in: node.inShapes, out: node.outShape });
    } catch {
      explanation = "";
    }
  }
  const block = node.templateId ? ir.groups[node.groupId] : undefined;
  const params = Object.entries(node.params).filter(
    ([key, value]) => value !== undefined && key !== "description",
  );

  return (
    <>
      <Section title={spec?.title ?? node.op}>
        <p className="pb-1.5 text-[13px] font-semibold text-text-1">{node.label}</p>
        <div className="flex flex-wrap items-center gap-1.5 pb-1.5">
          <Chip color={`var(${FAMILY_COLOR_TOKEN[spec?.family ?? "custom"]})`}>
            {spec?.family ?? "custom"}
          </Chip>
          <Chip color={status.color}>{status.label}</Chip>
          {node.wrapper ? <Chip>per item</Chip> : null}
          {node.trainable === "no" ? <Chip color="var(--frozen)">frozen</Chip> : null}
        </div>
        {explanation ? (
          <p className="pb-1 text-xs leading-relaxed text-text-2">{explanation}</p>
        ) : null}
        {block ? <p className="text-[11px] text-text-3">Part of {block.label}.</p> : null}
        {node.customOpName ? (
          <p className="text-[11px] text-text-3">
            Defined in the notebook as <span className="font-mono">{node.customOpName}</span>.
          </p>
        ) : null}
      </Section>

      <Section title="Shapes">
        {node.inShapes.length === 0 ? (
          <HudReadout label="Input" value="none (this is an input)" />
        ) : null}
        {node.inShapes.map((shape, index) => (
          <HudReadout
            key={index}
            label={node.inShapes.length > 1 ? `Input ${index + 1}` : "Input"}
            value={shapeText(shape)}
          />
        ))}
        <HudReadout
          label="Output"
          value={node.outShape ? shapeText(node.outShape) : "unknown"}
          source={trust}
        />
        <p className="pt-0.5 text-[11px] text-text-3">B is the batch size.</p>
      </Section>

      <Section title="Weights">
        <HudReadout label="Params" value={formatCount(total)} source={trust} />
        {total > 0 ? (
          <>
            <HudReadout label="Trainable" value={formatCount(node.paramCount.trainable)} />
            <HudReadout label="Non-trainable" value={formatCount(node.paramCount.nonTrainable)} />
          </>
        ) : null}
        <HudReadout label="FLOPs per sample" value={formatCompact(node.flops)} source="computed" />
        {row ? (
          <p className="pt-1 text-[11px] leading-snug text-text-3">
            Notebook summary row <span className="font-mono">{row.name || row.type}</span>:{" "}
            {row.shapeText || "no shape"}, {row.params === null ? "?" : formatCount(row.params)}{" "}
            params.
          </p>
        ) : null}
      </Section>

      {params.length > 0 ? (
        <Section title="Settings">
          {params.map(([key, value]) => (
            <HudReadout
              key={key}
              label={key}
              value={paramText(value)}
              className="[&>span:first-child]:font-mono [&>span:first-child]:text-[11px]"
            />
          ))}
        </Section>
      ) : null}

      {node.issues.length > 0 ? (
        <Section title="Notes">
          <p className="pb-1 text-xs leading-relaxed text-text-3">{status.text}</p>
          {node.issues.map((issue) => (
            <p
              key={issue.code + issue.message}
              className={`pb-1 text-xs leading-relaxed ${issue.severity === "error" ? "text-error" : issue.severity === "warn" ? "text-warn" : "text-text-3"}`}
            >
              {issue.message}
            </p>
          ))}
        </Section>
      ) : null}

      <CodeRefs refs={node.refs} />
    </>
  );
}

function GroupDetails({ ir, group }: { ir: RuntimeIR; group: RuntimeGroup }) {
  const total = group.stats.params.trainable + group.stats.params.nonTrainable;
  const members = ir.index.children[group.id];
  return (
    <>
      <Section title={`Group · ${group.kind}`}>
        <p className="pb-1.5 text-[13px] font-semibold text-text-1">{group.label}</p>
        {group.description ? (
          <p className="pb-1 text-xs leading-relaxed text-text-2">{group.description}</p>
        ) : null}
        <HudReadout label="Layers" value={formatCount(group.stats.nodeCount)} />
        <HudReadout label="Params" value={formatCount(total)} source="computed" />
        <HudReadout label="Trainable" value={formatCount(group.stats.params.trainable)} />
        <HudReadout
          label="FLOPs per sample"
          value={formatCompact(group.stats.flops)}
          source="computed"
        />
        <button
          type="button"
          onClick={() => useModelStore.getState().toggleGroup(group.id)}
          className="mt-1.5 text-xs text-accent hover:underline"
        >
          Open this group
        </button>
      </Section>
      {members && members.nodes.length > 0 ? (
        <Section title="Layers inside">
          {members.nodes.map((id) => (
            <p key={id} className="truncate py-0.5 text-xs text-text-2">
              {ir.nodes[id]!.label}
            </p>
          ))}
        </Section>
      ) : null}
      <CodeRefs refs={group.refs} />
    </>
  );
}

const ROLE_TEXT: Record<string, string> = {
  declare: "created",
  apply: "applied",
  definition: "defined",
  call: "called",
  loop: "repeated",
  config: "configured",
};

/** Where this layer comes from in the notebook. Only shown when a notebook is open. */
function CodeRefs({ refs }: { refs: RuntimeNode["refs"] }) {
  const hasNotebook = useProjectStore((state) => state.project !== null);
  const isListing = useModelStore((state) => state.listing !== null);
  if (refs.length === 0 || !hasNotebook || isListing) return null;
  return (
    <Section title="Code">
      {refs.map((ref) => (
        <button
          key={`${ref.cell}-${ref.startLine}-${ref.role}`}
          type="button"
          onClick={() => useProjectStore.getState().selectCell(ref.cell, "map")}
          className="flex w-full items-baseline justify-between gap-2 py-0.5 text-left text-xs text-text-2 hover:text-accent"
        >
          <span className="font-mono">
            cell {ref.cell} ·{" "}
            {ref.startLine === ref.endLine
              ? `line ${ref.startLine}`
              : `lines ${ref.startLine}–${ref.endLine}`}
          </span>
          <span className="text-text-3">{ROLE_TEXT[ref.role] ?? ref.role}</span>
        </button>
      ))}
    </Section>
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
