import { z } from "zod";
import type { ExtractEvent, ExtractRequest } from "@/core/ai/events";
import { runExtraction } from "@/server/ai/extract";
import { PROVIDER_IDS } from "@/server/ai/providers";

/** Large notebooks and repair rounds take a while on free tiers. */
export const maxDuration = 300;

const bodySchema = z.object({
  provider: z.enum(PROVIDER_IDS as [string, ...string[]]),
  model: z.string().min(1).max(200),
  context: z.string().min(1).max(400_000),
  summaries: z.array(z.unknown()).max(50).default([]),
  mode: z.enum(["extract", "design"]).default("extract"),
});

/**
 * Streams the extraction as newline-delimited JSON events (ExtractEvent).
 * The API key is added on the server; the browser only sends the code slice.
 */
export async function POST(request: Request): Promise<Response> {
  const parsed = bodySchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    const event: ExtractEvent = {
      type: "error",
      code: "bad_request",
      message: "The request was not understood.",
    };
    return Response.json(event, { status: 400 });
  }
  const extract = parsed.data as unknown as ExtractRequest;
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const event of runExtraction(extract, request.signal)) {
          controller.enqueue(encoder.encode(JSON.stringify(event) + "\n"));
        }
      } catch {
        const event: ExtractEvent = {
          type: "error",
          code: "provider",
          message: "The analysis stopped unexpectedly.",
        };
        controller.enqueue(encoder.encode(JSON.stringify(event) + "\n"));
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Accel-Buffering": "no",
    },
  });
}
