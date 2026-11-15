/**
 * The SimPlan: the forward pass as a list of timed steps, one per block on
 * screen. A collapsed group is a single, longer step. `stateAt` is a pure
 * function of time, so scrubbing is exact and repeatable.
 */
import type { VisibleGraph } from "../layout/graph";
import type { OpFamily } from "../ops/kinds";

export type SimStep = {
  index: number;
  /** The block this step lights up: a node id, or the id of a collapsed group. */
  itemId: string;
  /** The layers computed during this step. */
  memberIds: string[];
  label: string;
  family: OpFamily;
  /** Blocks whose output flows into this one. */
  sources: string[];
  start: number;
  duration: number;
};

export type SimPlan = { steps: SimStep[]; total: number };

/**
 * Within a step: data travels along the incoming edges, the layer works, then
 * its output appears. The fractions are of the step's duration.
 */
export type SimPhase = "travel" | "compute" | "emit";
const TRAVEL_END = 0.34;
const COMPUTE_END = 0.8;

export type SimState = {
  step: number;
  /** 0..1 through the current step. */
  progress: number;
  phase: SimPhase;
  /** 0..1 through the current phase. */
  phaseProgress: number;
  /** True once the last step has finished. */
  ended: boolean;
};

const BASE_SECONDS = 1.1;

export function buildPlan(graph: VisibleGraph): SimPlan {
  const sources = new Map<string, string[]>();
  for (const edge of graph.edges) {
    (sources.get(edge.to) ?? sources.set(edge.to, []).get(edge.to)!).push(edge.from);
  }
  let clock = 0;
  const steps = graph.items.map((item, index): SimStep => {
    // Inputs just appear; a collapsed group gets more time, growing slowly with its size.
    const duration =
      item.op === "input"
        ? BASE_SECONDS * 0.6
        : item.kind === "group"
          ? BASE_SECONDS * (1.2 + Math.log2(item.memberIds.length + 1) * 0.3)
          : BASE_SECONDS;
    const step: SimStep = {
      index,
      itemId: item.id,
      memberIds: item.memberIds,
      label: item.label,
      family: item.family,
      sources: [...new Set(sources.get(item.id) ?? [])],
      start: clock,
      duration,
    };
    clock += duration;
    return step;
  });
  return { steps, total: clock };
}

export function stateAt(plan: SimPlan, time: number): SimState {
  const last = plan.steps.length - 1;
  if (last < 0) return { step: -1, progress: 0, phase: "travel", phaseProgress: 0, ended: true };
  if (time >= plan.total)
    return { step: last, progress: 1, phase: "emit", phaseProgress: 1, ended: true };
  const t = Math.max(0, time);
  // Binary search for the step that contains t.
  let low = 0;
  let high = last;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (plan.steps[middle]!.start <= t) low = middle;
    else high = middle - 1;
  }
  const step = plan.steps[low]!;
  const progress = Math.min(1, (t - step.start) / step.duration);
  const [phase, from, to]: [SimPhase, number, number] =
    progress < TRAVEL_END
      ? ["travel", 0, TRAVEL_END]
      : progress < COMPUTE_END
        ? ["compute", TRAVEL_END, COMPUTE_END]
        : ["emit", COMPUTE_END, 1];
  return {
    step: low,
    progress,
    phase,
    phaseProgress: (progress - from) / (to - from),
    ended: false,
  };
}

/** The moment a step begins, used by "previous / next step". */
export const timeOfStep = (plan: SimPlan, index: number): number =>
  plan.steps[Math.max(0, Math.min(plan.steps.length - 1, index))]?.start ?? 0;

/**
 * The plan for a Lens focus: only the steps that compute a focused layer,
 * played back to back from time 0. The rest of the model is still computed
 * (the focus needs real inputs), it is just not stepped through.
 */
export function focusPlan(plan: SimPlan, nodeIds: readonly string[]): SimPlan {
  const wanted = new Set(nodeIds);
  let clock = 0;
  const steps = plan.steps
    .filter((step) => step.memberIds.some((id) => wanted.has(id)))
    .map((step, index): SimStep => {
      const moved = { ...step, index, start: clock };
      clock += step.duration;
      return moved;
    });
  return { steps, total: clock };
}
