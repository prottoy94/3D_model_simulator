import "server-only";
import type { AiErrorCode, ProviderId, ProviderInfo } from "@/core/ai/events";

/**
 * AI providers (ADR-013). All four speak the OpenAI-compatible chat API, so
 * one client serves them; what differs is the address, the key and which
 * models are worth offering. Keys are read here and never leave the server.
 */

type RawModel = {
  id: string;
  context_window?: number;
  context_length?: number;
  pricing?: { prompt?: string; completion?: string };
  supported_parameters?: string[];
};

type ProviderConfig = {
  id: ProviderId;
  label: string;
  baseUrl: () => string | undefined;
  apiKey: () => string | undefined;
  /** Picks and orders the chat models worth offering. */
  pick: (models: RawModel[]) => RawModel[];
  /** Tried in order for the default selection. */
  preferred: string[];
  /** Offered when the provider's model list cannot be loaded. */
  fallback: string[];
  dataNote: string;
};

const clean = (value: string | undefined) => value?.trim().replace(/^"|"$/g, "") || undefined;
const env = (name: string) => clean(process.env[name]);

const FREE_TIER_NOTE =
  "On a free tier the provider may use what you send to improve its models. Check its data terms before sending unpublished work.";

const CONFIG: Record<ProviderId, ProviderConfig> = {
  gemini: {
    id: "gemini",
    label: "Gemini",
    baseUrl: () => "https://generativelanguage.googleapis.com/v1beta/openai",
    apiKey: () => env("GOOGLE_API_KEY"),
    pick: (models) =>
      models
        .map((model) => ({ ...model, id: model.id.replace(/^models\//, "") }))
        .filter((model) =>
          /^gemini-[\d.]+-(flash|pro)(-lite)?$|^gemini-(flash|pro|flash-lite)-latest$/.test(
            model.id,
          ),
        )
        .sort((a, b) => b.id.localeCompare(a.id, undefined, { numeric: true })),
    // The newest Gemini models are often overloaded on the free tier; 2.5 Flash answered reliably when tested.
    preferred: ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest"],
    fallback: ["gemini-2.5-flash"],
    dataNote: FREE_TIER_NOTE,
  },
  groq: {
    id: "groq",
    label: "Groq",
    baseUrl: () => "https://api.groq.com/openai/v1",
    apiKey: () => env("GROQ_API_KEY"),
    pick: (models) =>
      models.filter(
        (model) =>
          (model.context_window ?? 0) >= 32_000 && !/whisper|guard|orpheus|tts/i.test(model.id),
      ),
    preferred: ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
    fallback: ["openai/gpt-oss-120b"],
    dataNote:
      "Groq's free tier has tight per-minute token limits, so a long notebook may be refused. " +
      FREE_TIER_NOTE,
  },
  qwen: {
    id: "qwen",
    label: "Qwen",
    baseUrl: () => env("DASHSCOPE_BASE_URL"),
    apiKey: () => env("DASHSCOPE_API_KEY"),
    pick: (models) =>
      models.filter(
        (model) =>
          /^qwen/i.test(model.id) && !/vl|audio|tts|asr|image|embedding|mt/i.test(model.id),
      ),
    preferred: ["qwen3.8-max", "qwen3.8-flash"],
    fallback: ["qwen3.8-max", "qwen3.8-flash"],
    dataNote:
      "Your Qwen quota is 1 million free tokens; usage after that is billed. " + FREE_TIER_NOTE,
  },
  openrouter: {
    id: "openrouter",
    label: "OpenRouter (free models)",
    baseUrl: () => "https://openrouter.ai/api/v1",
    apiKey: () => env("OPENROUTER_API_KEY"),
    // Free models only: a ":free" id AND a zero price, with room for a whole notebook.
    pick: (models) =>
      models
        .filter(
          (model) =>
            model.id.endsWith(":free") &&
            Number(model.pricing?.prompt ?? "1") === 0 &&
            Number(model.pricing?.completion ?? "1") === 0 &&
            (model.context_length ?? 0) >= 100_000,
        )
        .sort(
          (a, b) =>
            Number(b.supported_parameters?.includes("structured_outputs") ?? false) -
              Number(a.supported_parameters?.includes("structured_outputs") ?? false) ||
            (b.context_length ?? 0) - (a.context_length ?? 0),
        ),
    preferred: [],
    fallback: [],
    dataNote:
      "Only models that cost nothing are listed. Free models are often rate limited and change without notice. " +
      FREE_TIER_NOTE,
  },
};

export const PROVIDER_IDS = Object.keys(CONFIG) as ProviderId[];

export class AiError extends Error {
  constructor(
    public readonly code: AiErrorCode,
    message: string,
  ) {
    super(message);
    this.name = "AiError";
  }
}

function errorFor(provider: ProviderConfig, status: number, body: string): AiError {
  let detail = body.slice(0, 300);
  try {
    const json = JSON.parse(body) as
      { error?: { message?: string } | string } | Array<{ error?: { message?: string } }>;
    const error = Array.isArray(json) ? json[0]?.error : json.error;
    const message = typeof error === "string" ? error : error?.message;
    if (message) detail = message.slice(0, 300);
  } catch {
    // Not JSON: keep the raw text.
  }
  if (status === 401 || status === 403) {
    return new AiError(
      "auth",
      `${provider.label} rejected the API key. Check it in .env.local. (${detail})`,
    );
  }
  if (status === 429) {
    return new AiError(
      "rate_limit",
      `${provider.label} is rate limiting this key or the free quota is used up. Wait a minute or switch provider. (${detail})`,
    );
  }
  if (status === 503 || status === 529) {
    return new AiError(
      "rate_limit",
      `${provider.label} says this model is overloaded right now. Try again shortly or pick another model. (${detail})`,
    );
  }
  if (status === 404) {
    return new AiError(
      "provider",
      `This model is not available for your ${provider.label} key. Pick another model. (${detail})`,
    );
  }
  if (status === 413 || /context|too (large|long)|token/i.test(detail)) {
    return new AiError(
      "too_large",
      `The request is too large for this ${provider.label} model. Pick a model with a longer context. (${detail})`,
    );
  }
  return new AiError("provider", `${provider.label} returned an error (HTTP ${status}). ${detail}`);
}

function settings(id: ProviderId): { config: ProviderConfig; baseUrl: string; apiKey: string } {
  const config = CONFIG[id];
  const baseUrl = config?.baseUrl();
  const apiKey = config?.apiKey();
  if (!config || !baseUrl || !apiKey) {
    throw new AiError(
      "not_configured",
      `No API key is set for ${config?.label ?? id}. Add it to .env.local and restart the dev server.`,
    );
  }
  return { config, baseUrl: baseUrl.replace(/\/$/, ""), apiKey };
}

// ── Model lists ───────────────────────────────────────────────────────

const LIST_TTL_MS = 10 * 60 * 1000;
const listCache = new Map<ProviderId, { at: number; info: ProviderInfo }>();

async function describe(id: ProviderId): Promise<ProviderInfo> {
  const config = CONFIG[id];
  const base: ProviderInfo = {
    id,
    label: config.label,
    configured: Boolean(config.apiKey() && config.baseUrl()),
    models: [],
    defaultModel: null,
    dataNote: config.dataNote,
    error: null,
  };
  if (!base.configured) return base;

  let ids = config.fallback;
  try {
    const { baseUrl, apiKey } = settings(id);
    const response = await fetch(`${baseUrl}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(12_000),
    });
    if (!response.ok) throw errorFor(config, response.status, await response.text());
    const json = (await response.json()) as { data?: RawModel[] };
    const picked = config.pick(json.data ?? []).map((model) => model.id);
    if (picked.length > 0) ids = picked;
  } catch (error) {
    base.error = error instanceof Error ? error.message : "The model list could not be loaded.";
  }
  base.models = ids.map((model) => ({ id: model, label: model.replace(/:free$/, "") }));
  base.defaultModel = config.preferred.find((model) => ids.includes(model)) ?? ids[0] ?? null;
  return base;
}

export async function listProviders(): Promise<ProviderInfo[]> {
  return Promise.all(
    PROVIDER_IDS.map(async (id) => {
      const cached = listCache.get(id);
      if (cached && Date.now() - cached.at < LIST_TTL_MS) return cached.info;
      const info = await describe(id);
      // A failed list is not cached, so fixing the key takes effect on the next try.
      if (!info.error) listCache.set(id, { at: Date.now(), info });
      return info;
    }),
  );
}

export function defaultProvider(): ProviderId {
  const wanted = env("AI_PROVIDER") as ProviderId | undefined;
  return wanted && PROVIDER_IDS.includes(wanted) ? wanted : "gemini";
}

// ── Chat ──────────────────────────────────────────────────────────────

/** How long a model may take to start answering, and to continue once it has started. */
const START_TIMEOUT_MS = 75_000;
const STALL_TIMEOUT_MS = 60_000;

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type ChatResult = { text: string; usage: { input: number; output: number } | null };

type StreamChunk = {
  choices?: Array<{
    delta?: {
      content?: string | null;
      reasoning?: string | null;
      reasoning_content?: string | null;
    };
  }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number } | null;
  error?: { message?: string };
};

/**
 * One streamed chat completion that must answer in JSON. `onDelta` receives
 * the reply (and any visible reasoning) as it arrives.
 */
export async function chatJson(
  id: ProviderId,
  model: string,
  messages: ChatMessage[],
  onDelta: (kind: "text" | "reasoning", text: string) => void,
  signal?: AbortSignal,
): Promise<ChatResult> {
  const { config, baseUrl, apiKey } = settings(id);
  const body: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    temperature: 0.1,
    response_format: { type: "json_object" },
    // Asks for token counts at the end of the stream.
    stream_options: { include_usage: true },
  };
  // A model that never answers must not hang the app: give up if the reply does not
  // start, or stalls, for too long. Overloaded free models do this regularly.
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort);
  let timedOut = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const watchdog = (ms: number) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, ms);
  };
  const busy = () =>
    new AiError(
      "rate_limit",
      `${config.label} did not answer in time with ${model}. The model is probably overloaded: try again or pick another model.`,
    );

  const send = (payload: Record<string, unknown>) =>
    fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        // OpenRouter shows these on its dashboard; they carry nothing about the user.
        ...(id === "openrouter" ? { "X-Title": "NeuroScope" } : {}),
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

  let text = "";
  let usage: ChatResult["usage"] = null;
  try {
    watchdog(START_TIMEOUT_MS);
    let response = await send(body);
    // "Overloaded" usually clears within seconds on free tiers, so try twice more before giving up.
    for (const waitMs of [3_000, 8_000]) {
      if (response.status !== 503 && response.status !== 529) break;
      await response.body?.cancel();
      watchdog(waitMs + START_TIMEOUT_MS);
      await new Promise((resolve) => setTimeout(resolve, waitMs));
      response = await send(body);
    }
    if (response.status === 400) {
      // Some models reject JSON mode; the prompt already demands JSON, so retry without it.
      const text = await response.text();
      if (/response_format|json|stream_options|include_usage/i.test(text)) {
        // Fall back to the plainest request every OpenAI-compatible server accepts.
        const { response_format: _format, stream_options: _usage, ...rest } = body;
        void _format;
        void _usage;
        response = await send(rest);
      } else throw errorFor(config, 400, text);
    }
    if (!response.ok || !response.body)
      throw errorFor(config, response.status, await response.text());

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      watchdog(STALL_TIMEOUT_MS);
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const data = line.startsWith("data:") ? line.slice(5).trim() : "";
        if (!data || data === "[DONE]") continue;
        let chunk: StreamChunk;
        try {
          chunk = JSON.parse(data) as StreamChunk;
        } catch {
          continue; // keep-alive comments and partial lines
        }
        if (chunk.error?.message)
          throw new AiError("provider", `${config.label}: ${chunk.error.message.slice(0, 300)}`);
        const delta = chunk.choices?.[0]?.delta;
        const reasoning = delta?.reasoning ?? delta?.reasoning_content;
        if (reasoning) onDelta("reasoning", reasoning);
        if (delta?.content) {
          text += delta.content;
          onDelta("text", delta.content);
        }
        if (chunk.usage) {
          usage = {
            input: chunk.usage.prompt_tokens ?? 0,
            output: chunk.usage.completion_tokens ?? 0,
          };
        }
      }
    }
  } catch (error) {
    if (timedOut) throw busy();
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
  if (!text.trim())
    throw new AiError(
      "invalid_reply",
      `${config.label} returned an empty reply. Try again or pick another model.`,
    );
  return { text, usage };
}
