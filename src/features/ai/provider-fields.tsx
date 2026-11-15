"use client";

import type { ProviderId } from "@/core/ai/events";
import { currentModel, useAiStore } from "@/stores/ai-store";

const selectClass =
  "h-8 w-full min-w-0 rounded-control border border-line bg-glass-2 px-2 font-mono text-xs text-text-1 outline-none focus-visible:border-accent disabled:opacity-50";

/** The provider and model pickers, shared by the top-bar menu and the Analyze panel. */
export function ProviderFields({ disabled = false }: { disabled?: boolean }) {
  const providers = useAiStore((state) => state.providers);
  const provider = useAiStore((state) => state.provider);
  const model = useAiStore((state) => currentModel(state));
  const setProvider = useAiStore((state) => state.setProvider);
  const setModel = useAiStore((state) => state.setModel);

  if (providers === null) return <p className="text-xs text-text-3">Loading providers…</p>;
  const usable = providers.filter((entry) => entry.configured);
  if (usable.length === 0) {
    return (
      <p className="text-xs leading-relaxed text-warn">
        No AI key is set. Add one to <span className="font-mono">.env.local</span> (see{" "}
        <span className="font-mono">.env.example</span>) and restart the dev server.
      </p>
    );
  }
  const current = usable.find((entry) => entry.id === provider);

  return (
    <div className="flex flex-col gap-2">
      <label className="flex flex-col gap-1 text-[11px] text-text-3">
        Provider
        <select
          className={selectClass}
          value={provider ?? ""}
          disabled={disabled}
          onChange={(event) => setProvider(event.target.value as ProviderId)}
        >
          {providers.map((entry) => (
            <option key={entry.id} value={entry.id} disabled={!entry.configured}>
              {entry.label}
              {entry.configured ? "" : " (no key)"}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1 text-[11px] text-text-3">
        Model
        <select
          className={selectClass}
          value={model ?? ""}
          disabled={disabled || !current || current.models.length === 0}
          onChange={(event) => setModel(event.target.value)}
        >
          {(current?.models ?? []).map((entry) => (
            <option key={entry.id} value={entry.id}>
              {entry.label}
            </option>
          ))}
        </select>
      </label>
      {current?.error ? (
        <p className="text-[11px] leading-snug text-warn">
          The live model list could not be loaded, so a short default list is shown. {current.error}
        </p>
      ) : null}
    </div>
  );
}
