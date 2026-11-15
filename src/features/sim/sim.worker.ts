/**
 * Runs the forward pass off the main thread, so the 3D view stays smooth while
 * a large model computes. It posts one small summary per layer as it finishes;
 * the full tensors stay here and are freed as soon as nothing needs them.
 * Stopping a run means terminating this worker.
 */
import type { RuntimeIR } from "@/core/ir";
import { simulate, type SimInput } from "@/core/sim/run";
import { summarize, transferables, type TensorView } from "@/core/sim/view";

export type SimRequest = { ir: RuntimeIR; modelId: string; input: SimInput };

export type SimMessage =
  | { type: "view"; view: TensorView; done: number; total: number }
  | { type: "finished"; ms: number }
  | { type: "failed"; message: string };

type WorkerScope = {
  onmessage: ((event: MessageEvent<SimRequest>) => void) | null;
  postMessage(message: SimMessage, transfer?: Transferable[]): void;
};
const scope = self as unknown as WorkerScope;

scope.onmessage = (event) => {
  const { ir, modelId, input } = event.data;
  const started = performance.now();
  const total = ir.index.topoOrder[modelId]?.length ?? 0;
  let done = 0;
  try {
    for (const result of simulate(ir, modelId, input)) {
      const view = summarize(result, ir.nodes[result.nodeId]?.op === "input");
      done++;
      scope.postMessage({ type: "view", view, done, total }, transferables(view) as Transferable[]);
    }
    scope.postMessage({ type: "finished", ms: performance.now() - started });
  } catch (error) {
    scope.postMessage({
      type: "failed",
      message: error instanceof Error ? error.message : "The simulation stopped unexpectedly.",
    });
  }
};
