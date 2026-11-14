"use client";

import { Chip, HudReadout } from "@/components/ui";
import { formatCount } from "@/core/format";
import { formatStaticValue } from "@/core/python";
import { ROLE_META } from "@/features/notebook-map/roles";
import { useProjectStore } from "@/stores/project-store";

const megabytes = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

/**
 * Phase 1 inspector: what the deterministic analysis knows about the notebook
 * and the selected cell. Layer-level inspection replaces this in Phase 3.
 */
export function NotebookInspector() {
  const project = useProjectStore((state) => state.project);
  const selectedCell = useProjectStore((state) => state.selectedCell);
  if (!project) return null;

  const { analysis, notebook } = project;
  const cell = analysis.cells.find((entry) => entry.index === selectedCell) ?? null;
  const constants = Object.entries(analysis.constants);
  const models = project.ir?.models ?? [];

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden [overflow-wrap:anywhere]">
      <Section title="Notebook">
        <HudReadout label="Frameworks" value={analysis.frameworks.join(", ") || "unknown"} />
        <HudReadout
          label="Cells"
          value={`${notebook.stats.codeCells} code · ${notebook.stats.markdownCells} text`}
        />
        <HudReadout
          label="Model cells"
          value={analysis.modelCells.length > 0 ? analysis.modelCells.join(", ") : "none found"}
          source={analysis.modelCells.length > 0 ? "computed" : "mismatch"}
        />
        {analysis.classCounts.length > 0 ? (
          <HudReadout
            label="Classes (from outputs)"
            value={analysis.classCounts.join(" / ")}
            source="summary"
          />
        ) : null}
        <HudReadout
          label="Images removed"
          value={`${notebook.stats.droppedImages} · ${megabytes(notebook.stats.droppedBytes)}`}
        />
        <HudReadout label="Analyzed in" value={`${project.analysisMs} ms`} />
      </Section>

      <Section title="Models · read without AI">
        {models.length === 0 ? (
          <p className="pb-1 text-xs leading-relaxed text-text-3">
            No simple layer stack was found. This notebook needs the AI step (Phase 4) to build its
            architecture.
          </p>
        ) : null}
        {models.map((model) => {
          const report = project.ir!.verification.perModel[model.id];
          if (!report) return null;
          const { computed, match } = report.totals;
          const source = match === null ? "computed" : match ? "summary" : "mismatch";
          return (
            <div key={model.id} className="border-t border-line pt-1 first:border-t-0 first:pt-0">
              <p className="hud-label !text-text-1">{model.name}</p>
              <HudReadout
                label="Layers"
                value={formatCount(project.ir!.index.topoOrder[model.id]?.length ?? 0)}
              />
              <HudReadout
                label="Total params"
                value={formatCount(computed.total)}
                source={source}
              />
              <HudReadout
                label="Trainable"
                value={formatCount(computed.trainable)}
                source={source}
              />
              <HudReadout
                label="Non-trainable"
                value={formatCount(computed.nonTrainable)}
                source={source}
              />
              <p className="pb-1 text-xs leading-relaxed text-text-3">
                {match === null
                  ? "Computed from the code. The notebook has no saved summary to compare with."
                  : match
                    ? `Computed from the code and equal to the notebook's saved summary (${report.rows.length} layers compared).`
                    : "Computed from the code, but the notebook's saved summary shows different numbers."}
              </p>
            </div>
          );
        })}
        {project.declined.map((entry) => (
          <p
            key={`${entry.cell}-${entry.line}`}
            className="pb-1 text-xs leading-relaxed text-text-3"
          >
            · Cell {entry.cell}, line {entry.line}: skipped, {entry.reason}.
          </p>
        ))}
      </Section>

      {cell ? (
        <Section title={`Cell ${cell.index}`}>
          <div className="flex flex-wrap items-center gap-2 pb-1">
            <Chip color={ROLE_META[cell.role].color}>{ROLE_META[cell.role].label}</Chip>
            {cell.role !== "markdown" ? (
              <span className="text-xs text-text-3">{cell.confidence} confidence</span>
            ) : null}
          </div>
          <p className="pb-1 text-xs leading-relaxed text-text-2">
            {ROLE_META[cell.role].description}
          </p>
          {cell.signals.length > 0 ? (
            <ul className="pb-1 text-xs leading-relaxed text-text-3">
              {cell.signals.map((signal) => (
                <li key={signal}>· {signal}</li>
              ))}
            </ul>
          ) : null}
          {cell.analysis?.hasSyntaxError ? (
            <p className="pb-1 text-xs text-warn">
              This cell has syntax the parser could not read.
            </p>
          ) : null}

          {cell.summaries.map((summary, position) => (
            <div key={position} className="mt-1 border-t border-line pt-1">
              <p className="hud-label !text-text-1">
                Saved summary{summary.modelName ? ` · ${summary.modelName}` : ""}
              </p>
              <HudReadout label="Layers listed" value={formatCount(summary.rows.length)} />
              {summary.totals.total !== null ? (
                <HudReadout
                  label="Total params"
                  value={formatCount(summary.totals.total)}
                  source="summary"
                />
              ) : null}
              {summary.totals.trainable !== null ? (
                <HudReadout
                  label="Trainable"
                  value={formatCount(summary.totals.trainable)}
                  source="summary"
                />
              ) : null}
              {summary.totals.nonTrainable !== null ? (
                <HudReadout
                  label="Non-trainable"
                  value={formatCount(summary.totals.nonTrainable)}
                  source="summary"
                />
              ) : null}
            </div>
          ))}

          {cell.facts.paramMentions.map((mention) => (
            <HudReadout
              key={`${mention.label}-${mention.value}`}
              label={mention.label || "Printed params"}
              value={`${mention.approximate ? "≈ " : ""}${formatCount(mention.value)}`}
              source="summary"
            />
          ))}
        </Section>
      ) : null}

      {constants.length > 0 ? (
        <Section title={`Constants · ${constants.length}`}>
          {constants.map(([name, value]) => (
            <HudReadout
              key={name}
              label={name}
              value={truncate(formatStaticValue(value), 26)}
              source="computed"
              className="[&>span:first-child]:font-mono [&>span:first-child]:text-[11px]"
            />
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

function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
