import "server-only";
import { describeProblems } from "@/core/ai/context";
import {
  applyDraftPatch,
  draftToWire,
  parseDraft,
  parseDraftPatch,
  type Draft,
} from "@/core/ai/draft";
import type { ExtractEvent, ExtractRequest } from "@/core/ai/events";
import { designAddendum, invalidReplyPrompt, repairPrompt, systemPrompt } from "@/core/ai/prompt";
import { buildRuntimeIR } from "@/core/ir/build";
import { AiError, chatJson, type ChatMessage } from "./providers";

/** At most this many follow-up requests after the first one (doc 03 §6). */
const MAX_REPAIR_ROUNDS = 2;

/** What the engine thinks of a draft. An empty list means it found nothing wrong. */
function check(draft: Draft, request: ExtractRequest): string[] {
  const built = buildRuntimeIR(draftToWire(draft), { summaries: request.summaries });
  // A designed model has no code, so missing line numbers are not a problem.
  return built.ok
    ? describeProblems(built.ir, 14, request.mode !== "design")
    : built.errors.slice(0, 10);
}

/**
 * The whole "LLM proposes, engine verifies" loop for one notebook:
 * ask for a draft, check it with the shape engine against the saved
 * summaries, and send the differences back until it is clean or the rounds
 * run out. The conversation is append-only. Progress is yielded as events.
 */
export async function* runExtraction(
  request: ExtractRequest,
  signal?: AbortSignal,
): AsyncGenerator<ExtractEvent> {
  const messages: ChatMessage[] = [
    {
      role: "system",
      content: systemPrompt() + (request.mode === "design" ? designAddendum() : ""),
    },
    { role: "user", content: request.context },
  ];
  const usage = { input: 0, output: 0 };
  let sawUsage = false;
  let draft: Draft | null = null;
  let problems: string[] = [];
  let rounds = 0;

  // Deltas arrive inside chatJson; they are queued and yielded between awaits.
  const ask = async function* (stage: "extract" | "repair", message: string, details?: string[]) {
    yield { type: "stage", stage, message, details } satisfies ExtractEvent;
    const queue: ExtractEvent[] = [];
    let finished = false;
    let wake: (() => void) | null = null;
    const pending = chatJson(
      request.provider,
      request.model,
      messages,
      (kind, text) => {
        queue.push({ type: "delta", kind, text });
        wake?.();
      },
      signal,
    ).finally(() => {
      finished = true;
      wake?.();
    });
    while (!finished || queue.length > 0) {
      if (queue.length === 0) {
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
        wake = null;
        continue;
      }
      yield queue.shift()!;
    }
    const reply = await pending;
    if (reply.usage) {
      sawUsage = true;
      usage.input += reply.usage.input;
      usage.output += reply.usage.output;
    }
    messages.push({ role: "assistant", content: reply.text });
    return reply.text;
  };

  try {
    let text = yield* ask(
      "extract",
      request.mode === "design"
        ? `Designing the model with ${request.model}…`
        : `Reading the code with ${request.model}…`,
    );
    for (;;) {
      if (draft === null) {
        const parsed = parseDraft(text);
        if (parsed.ok) draft = parsed.value;
        else problems = parsed.errors;
      } else {
        const parsed = parseDraftPatch(text);
        if (parsed.ok) draft = applyDraftPatch(draft, parsed.value);
        // An unusable patch leaves the draft as it was; the same problems are reported again.
      }

      if (draft !== null) {
        yield {
          type: "stage",
          stage: "verify",
          message: "Recomputing every shape and parameter count…",
        };
        problems = check(draft, request);
        if (problems.length === 0) break;
      }
      if (rounds >= MAX_REPAIR_ROUNDS) break;
      rounds++;
      messages.push({
        role: "user",
        content: draft === null ? invalidReplyPrompt(problems) : repairPrompt(problems),
      });
      text = yield* ask(
        "repair",
        draft === null
          ? `The reply was not valid JSON. Asking again (round ${rounds} of ${MAX_REPAIR_ROUNDS})…`
          : `The engine found ${problems.length} problem${problems.length === 1 ? "" : "s"}. Asking for a fix (round ${rounds} of ${MAX_REPAIR_ROUNDS})…`,
        problems,
      );
    }

    if (draft === null) {
      yield {
        type: "error",
        code: "invalid_reply",
        message: `The model did not return usable JSON after ${rounds + 1} tries (${problems[0] ?? "unknown reason"}). Try another model.`,
      };
      return;
    }
    yield {
      type: "stage",
      stage: "done",
      message:
        problems.length === 0
          ? "Verified: nothing left to fix."
          : `${problems.length} problem${problems.length === 1 ? "" : "s"} could not be fixed and are flagged in the view.`,
    };
    yield { type: "result", draft, rounds, problems, usage: sawUsage ? usage : null };
  } catch (error) {
    if (error instanceof AiError) yield { type: "error", code: error.code, message: error.message };
    else if (signal?.aborted)
      yield { type: "error", code: "provider", message: "The request was cancelled." };
    else {
      yield {
        type: "error",
        code: "provider",
        message: `The request failed: ${error instanceof Error ? error.message : "unknown error"}`,
      };
    }
  }
}
