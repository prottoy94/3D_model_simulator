/** Types shared by the server's extract stream and the client that reads it. */
import type { ModelSummary } from "../notebook/summary";
import type { Draft } from "./draft";

export type ProviderId = "gemini" | "groq" | "qwen" | "openrouter";

export type ProviderInfo = {
  id: ProviderId;
  label: string;
  /** False when no API key is set for it on the server. */
  configured: boolean;
  models: Array<{ id: string; label: string }>;
  defaultModel: string | null;
  /** Shown before the first request to this provider. */
  dataNote: string;
  error: string | null;
};

export type ExtractRequest = {
  provider: ProviderId;
  model: string;
  /** The text from buildExtractionContext. */
  context: string;
  /** Saved summaries, so the server can verify and ask for repairs. */
  summaries: ModelSummary[];
  /** "design": `context` is a description of a model to invent, not notebook code. */
  mode?: "extract" | "design";
};

export type AiErrorCode =
  | "not_configured"
  | "auth"
  | "rate_limit"
  | "too_large"
  | "provider"
  | "invalid_reply"
  | "bad_request";

export type ExtractEvent =
  | {
      type: "stage";
      stage: "extract" | "verify" | "repair" | "done";
      message: string;
      /** For a repair: what the engine asked the model to fix. */
      details?: string[];
    }
  /** A piece of the model's reply (or of its reasoning), for the live log. */
  | { type: "delta"; kind: "text" | "reasoning"; text: string }
  | {
      type: "result";
      draft: Draft;
      rounds: number;
      problems: string[];
      usage: { input: number; output: number } | null;
    }
  | { type: "error"; code: AiErrorCode; message: string };
