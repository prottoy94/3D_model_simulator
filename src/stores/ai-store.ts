import { create } from "zustand";
import { buildExtractionContext } from "@/core/ai/context";
import type { ExtractEvent, ExtractRequest, ProviderId, ProviderInfo } from "@/core/ai/events";
import { useForgeStore } from "./forge-store";
import { useProjectStore } from "./project-store";

export type LogLine = { kind: "stage" | "detail" | "error" | "ok"; text: string };

type RunState = {
  status: "idle" | "running" | "done" | "error";
  log: LogLine[];
  /** Characters of model output received so far, and the last few of them. */
  received: number;
  tail: string;
  error: string | null;
  problems: string[];
  rounds: number;
  usage: { input: number; output: number } | null;
};

type Prefs = {
  provider: ProviderId | null;
  models: Partial<Record<ProviderId, string>>;
  /** Providers the user has agreed to send code to. */
  consent: Partial<Record<ProviderId, boolean>>;
  /** When on, nothing is ever sent to an AI provider. */
  localOnly: boolean;
};

type AiState = Prefs & {
  providers: ProviderInfo[] | null;
  loadError: string | null;
  /** null = decide automatically (open when a notebook has model code but no model on screen). */
  panelOpen: boolean | null;
  run: RunState;
  loadProviders: () => Promise<void>;
  setProvider: (id: ProviderId) => void;
  setModel: (model: string) => void;
  setLocalOnly: (on: boolean) => void;
  giveConsent: (id: ProviderId, given: boolean) => void;
  setPanelOpen: (open: boolean | null) => void;
  analyze: (description?: string) => Promise<void>;
  /** Forge: asks the provider to design a model from a description. */
  design: (description: string) => Promise<void>;
  cancel: () => void;
};

const STORAGE_KEY = "neuroscope.ai";
const IDLE: RunState = {
  status: "idle",
  log: [],
  received: 0,
  tail: "",
  error: null,
  problems: [],
  rounds: 0,
  usage: null,
};

function readPrefs(): Prefs {
  const empty: Prefs = { provider: null, models: {}, consent: {}, localOnly: false };
  if (typeof window === "undefined") return empty;
  try {
    return {
      ...empty,
      ...(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}") as Partial<Prefs>),
    };
  } catch {
    return empty;
  }
}

function writePrefs(state: Prefs): void {
  try {
    const { provider, models, consent, localOnly } = state;
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ provider, models, consent, localOnly }),
    );
  } catch {
    // Private mode: the choice just does not survive a reload.
  }
}

let controller: AbortController | null = null;

export const useAiStore = create<AiState>((set, get) => ({
  provider: null,
  models: {},
  consent: {},
  localOnly: false,
  providers: null,
  loadError: null,
  panelOpen: null,
  run: IDLE,

  loadProviders: async () => {
    try {
      const response = await fetch("/api/ai/providers");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = (await response.json()) as {
        providers: ProviderInfo[];
        defaultProvider: ProviderId;
      };
      const prefs = readPrefs();
      const usable = data.providers.filter((provider) => provider.configured);
      // Keep the saved choice when it still has a key; otherwise the server's default, then any.
      const provider =
        usable.find((entry) => entry.id === prefs.provider)?.id ??
        usable.find((entry) => entry.id === data.defaultProvider)?.id ??
        usable[0]?.id ??
        null;
      set({ ...prefs, provider, providers: data.providers, loadError: null });
    } catch {
      set({
        providers: [],
        loadError: "The AI settings could not be loaded from the app's server.",
      });
    }
  },

  setProvider: (provider) => {
    set({ provider });
    writePrefs(get());
  },
  setModel: (model) => {
    const { provider, models } = get();
    if (!provider) return;
    set({ models: { ...models, [provider]: model } });
    writePrefs(get());
  },
  setLocalOnly: (localOnly) => {
    set({ localOnly });
    writePrefs(get());
  },
  giveConsent: (id, given) => {
    set({ consent: { ...get().consent, [id]: given } });
    writePrefs(get());
  },
  setPanelOpen: (panelOpen) => set({ panelOpen }),

  cancel: () => controller?.abort(),

  design: async (description) => get().analyze(description),

  analyze: async (description?: string) => {
    const state = get();
    const project = useProjectStore.getState().project;
    const info = state.providers?.find((entry) => entry.id === state.provider);
    const model = currentModel(state);
    const designing = description !== undefined;
    if (!info || !model || state.localOnly || state.run.status === "running") return;
    if (!designing && !project) return;

    const context = designing ? null : buildExtractionContext(project!.notebook, project!.analysis);
    const body: ExtractRequest = {
      provider: info.id,
      model,
      context: designing ? description : context!.text,
      summaries: designing ? [] : project!.analysis.cells.flatMap((cell) => cell.summaries),
      mode: designing ? "design" : "extract",
    };
    set({
      // Stay open after the run, so the outcome can be read before closing.
      panelOpen: designing ? get().panelOpen : true,
      run: {
        ...IDLE,
        status: "running",
        log: [
          {
            kind: "stage",
            text: designing
              ? `Sending your description to ${info.label}.`
              : `Sending ${context!.cells.length} code cells (about ${context!.approxTokens.toLocaleString("en-US")} tokens) to ${info.label}.`,
          },
        ],
      },
    });
    const push = (line: LogLine) => set({ run: { ...get().run, log: [...get().run.log, line] } });
    const fail = (message: string) =>
      set({
        run: {
          ...get().run,
          status: "error",
          error: message,
          log: [...get().run.log, { kind: "error", text: message }],
        },
      });

    controller = new AbortController();
    // Output arrives in many small pieces; the view is refreshed a few times a second, not per piece.
    let received = 0;
    let tail = "";
    let flush: ReturnType<typeof setTimeout> | null = null;
    const scheduleFlush = () => {
      flush ??= setTimeout(() => {
        flush = null;
        set({ run: { ...get().run, received, tail } });
      }, 150);
    };

    const handle = (event: ExtractEvent) => {
      if (event.type === "delta") {
        received += event.text.length;
        tail = (tail + event.text).slice(-320);
        scheduleFlush();
      } else if (event.type === "stage") {
        push({ kind: event.stage === "done" ? "ok" : "stage", text: event.message });
        for (const detail of event.details ?? []) push({ kind: "detail", text: detail });
      } else if (event.type === "error") {
        fail(event.message);
      } else {
        if (designing) useForgeStore.getState().replace(event.draft);
        else useProjectStore.getState().applyDraft(event.draft, { provider: info.label, model });
        set({
          run: {
            ...get().run,
            status: "done",
            received,
            tail: "",
            problems: event.problems,
            rounds: event.rounds,
            usage: event.usage,
          },
        });
      }
    };

    try {
      const response = await fetch("/api/ai/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.body) throw new Error("no response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) if (line.trim()) handle(JSON.parse(line) as ExtractEvent);
      }
      if (buffer.trim()) handle(JSON.parse(buffer) as ExtractEvent);
      if (get().run.status === "running") fail("The analysis ended without a result. Try again.");
    } catch {
      if (get().run.status === "running") {
        fail(
          controller.signal.aborted ? "Cancelled." : "The connection to the app's server was lost.",
        );
      }
    } finally {
      if (flush) clearTimeout(flush);
      controller = null;
    }
  },
}));

/** The model in use for the selected provider: the saved choice if still offered, else the default. */
export function currentModel(
  state: Pick<AiState, "provider" | "providers" | "models">,
): string | null {
  const info = state.providers?.find((entry) => entry.id === state.provider);
  if (!info) return null;
  const saved = state.models[info.id];
  return saved && info.models.some((model) => model.id === saved) ? saved : info.defaultModel;
}
