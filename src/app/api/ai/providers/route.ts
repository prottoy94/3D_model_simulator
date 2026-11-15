import { connection } from "next/server";
import { defaultProvider, listProviders } from "@/server/ai/providers";

/** Which providers have a key, and the models each offers right now. Never returns a key. */
export async function GET(): Promise<Response> {
  // The answer depends on the server's environment and on live model lists, so it is never prerendered.
  await connection();
  const providers = await listProviders();
  return Response.json(
    { providers, defaultProvider: defaultProvider() },
    { headers: { "Cache-Control": "no-store" } },
  );
}
