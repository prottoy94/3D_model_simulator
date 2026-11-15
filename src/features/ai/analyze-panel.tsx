"use client";

import { useEffect, useMemo, useRef } from "react";
import { GlassPanel, NeonButton } from "@/components/ui";
import { buildExtractionContext } from "@/core/ai/context";
import { cn } from "@/lib/cn";
import { currentModel, useAiStore, type LogLine } from "@/stores/ai-store";
import { useProjectStore } from "@/stores/project-store";
import { ProviderFields } from "./provider-fields";

const lineClass: Record<LogLine["kind"], string> = {
  stage: "text-text-1",
  detail: "pl-4 text-text-3",
  error: "text-error",
  ok: "text-ok",
};
const linePrefix: Record<LogLine["kind"], string> = {
  stage: "›",
  detail: "·",
  error: "✕",
  ok: "✓",
};

/**
 * Sends the notebook's model code to the chosen AI provider. It always says
 * exactly what will leave the machine first, and nothing is sent before the
 * user has agreed for that provider.
 */
export function AnalyzePanel({ onClose }: { onClose: () => void }) {
  const project = useProjectStore((state) => state.project);
  const providers = useAiStore((state) => state.providers);
  const provider = useAiStore((state) => state.provider);
  const model = useAiStore((state) => currentModel(state));
  const consent = useAiStore((state) => state.consent);
  const localOnly = useAiStore((state) => state.localOnly);
  const run = useAiStore((state) => state.run);
  const { analyze, cancel, giveConsent } = useAiStore.getState();
  const logEnd = useRef<HTMLDivElement>(null);

  const context = useMemo(
    () => (project ? buildExtractionContext(project.notebook, project.analysis) : null),
    [project],
  );
  useEffect(() => {
    logEnd.current?.scrollIntoView({ block: "end" });
  }, [run.log.length, run.received]);

  if (!project || !context) return null;
  const info = providers?.find((entry) => entry.id === provider);
  const agreed = provider ? consent[provider] === true : false;
  const running = run.status === "running";
  const canSend =
    Boolean(info && model) && agreed && !localOnly && !running && context.cells.length > 0;

  return (
    <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center p-4">
      <GlassPanel
        raised
        role="dialog"
        aria-label="Analyze with AI"
        className="pointer-events-auto max-h-full w-full max-w-md gap-3 overflow-y-auto p-4"
      >
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="hud-label !text-accent">Analyze with AI</p>
            <p className="mt-1 text-sm font-semibold text-text-1">
              {project.draft ? "Analyze this notebook again" : "Build this notebook's architecture"}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="text-lg leading-none text-text-3 hover:text-text-1"
          >
            ×
          </button>
        </div>

        <div className="rounded-control border border-line bg-glass-1 p-2.5 text-xs leading-relaxed text-text-2">
          <p className="pb-1 font-semibold text-text-1">What will be sent</p>
          {context.cells.length > 0 ? (
            <>
              <p>
                The code of {context.cells.length} {context.cells.length === 1 ? "cell" : "cells"}{" "}
                that define or set up models (cells {context.cells.join(", ")}), the constants found
                in the notebook and its saved model summaries: about{" "}
                {context.approxTokens.toLocaleString("en-US")} tokens.
              </p>
              <p className="pt-1 text-text-3">
                Not sent: images, data, plots, and the other{" "}
                {project.analysis.cells.length - context.cells.length} cells.
                {context.truncated
                  ? " Some model cells were left out because the notebook is very long."
                  : ""}
              </p>
            </>
          ) : (
            <p className="text-warn">
              No cell in this notebook looks like it defines a model, so there is nothing to send.
            </p>
          )}
        </div>

        <ProviderFields disabled={running} />

        {localOnly ? (
          <p className="text-xs leading-relaxed text-warn">
            Local only is on, so nothing can be sent. Turn it off in the AI menu at the top right.
          </p>
        ) : info ? (
          <label className="flex cursor-pointer items-start gap-2 text-xs leading-relaxed text-text-2">
            <input
              type="checkbox"
              className="mt-0.5 accent-[var(--accent)]"
              checked={agreed}
              disabled={running}
              onChange={(event) => giveConsent(info.id, event.target.checked)}
            />
            <span>
              Send this code to <span className="font-semibold text-text-1">{info.label}</span>.{" "}
              <span className="text-text-3">{info.dataNote}</span>
            </span>
          </label>
        ) : null}

        {run.log.length > 0 ? (
          <div
            aria-live="polite"
            className="max-h-44 overflow-y-auto rounded-control border border-line bg-void p-2 font-mono text-[11px] leading-relaxed"
          >
            {run.log.map((line, index) => (
              <p
                key={index}
                className={cn("whitespace-pre-wrap [overflow-wrap:anywhere]", lineClass[line.kind])}
              >
                {linePrefix[line.kind]} {line.text}
              </p>
            ))}
            {running && run.received > 0 ? (
              <p className="text-text-3">
                … {run.received.toLocaleString("en-US")} characters received
                <span className="block truncate opacity-60">
                  {run.tail.replace(/\s+/g, " ").slice(-90)}
                </span>
              </p>
            ) : null}
            <div ref={logEnd} />
          </div>
        ) : null}

        {run.status === "done" ? (
          <p
            className={cn(
              "text-xs leading-relaxed",
              run.problems.length === 0 ? "text-ok" : "text-warn",
            )}
          >
            {run.problems.length === 0
              ? "Done. The engine recomputed every shape and count and found nothing wrong."
              : `Done, with ${run.problems.length} point${run.problems.length === 1 ? "" : "s"} the model could not fix. They are flagged in the view and listed in the inspector.`}{" "}
            <span className="text-text-3">
              {run.rounds > 0 ? `${run.rounds} repair round${run.rounds === 1 ? "" : "s"}. ` : ""}
              {run.usage
                ? `${run.usage.input.toLocaleString("en-US")} tokens in, ${run.usage.output.toLocaleString("en-US")} out. `
                : ""}
              The result is saved on this computer, so this notebook will not be sent again.
            </span>
          </p>
        ) : null}

        <div className="flex items-center justify-end gap-2">
          {running ? (
            <NeonButton size="sm" variant="danger" onClick={cancel}>
              Cancel
            </NeonButton>
          ) : (
            <>
              <NeonButton size="sm" onClick={onClose}>
                {run.status === "done" ? "Close" : "Not now"}
              </NeonButton>
              <NeonButton
                size="sm"
                variant="primary"
                disabled={!canSend}
                onClick={() => void analyze()}
              >
                {run.status === "done" || run.status === "error" ? "Analyze again" : "Analyze"}
              </NeonButton>
            </>
          )}
        </div>
      </GlassPanel>
    </div>
  );
}
