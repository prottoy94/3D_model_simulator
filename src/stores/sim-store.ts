import { create } from "zustand";
import { buildVisibleGraph } from "@/core/layout";
import { buildPlan, focusPlan, stateAt, timeOfStep, type SimPlan } from "@/core/sim/plan";
import { estimateWork, type SimInput } from "@/core/sim/run";
import type { TensorView } from "@/core/sim/view";
import type { SimMessage, SimRequest } from "@/features/sim/sim.worker";
import { useModelStore } from "./model-store";
import { useUiStore } from "./ui-store";

export type InputChoice = "shapes" | "stripes" | "noise" | "image";

/**
 * The simulation clock. It changes every frame, so it lives outside React
 * state: the 3D view and the timeline read it in their own animation loops.
 */
export const simClock = { time: 0 };

/** Measured on a laptop CPU: multiply-adds per second in the worker. Only used for the estimate. */
const WORK_PER_SECOND = 0.3e9;

type SimState = {
  plan: SimPlan | null;
  status: "idle" | "computing" | "ready" | "error";
  error: string | null;
  /** One summary per computed layer, keyed by node id. */
  views: Record<string, TensorView>;
  computed: number;
  total: number;
  /** Rough seconds a full run takes for the model on screen. */
  estimateSeconds: number;
  input: InputChoice;
  /** The user's own picture, when they supplied one. */
  image: Extract<SimInput, { kind: "image" }> | null;
  imageName: string | null;
  playing: boolean;
  /** True while playback is held because the next layer is still being computed. */
  waiting: boolean;
  speed: number;
  loop: boolean;
  follow: boolean;
  /** The step the clock is in. Changes about once a second, so React may subscribe to it. */
  step: number;
  ended: boolean;

  run: () => void;
  stop: () => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (time: number) => void;
  stepBy: (delta: number) => void;
  setSpeed: (speed: number) => void;
  setLoop: (loop: boolean) => void;
  setFollow: (follow: boolean) => void;
  setInput: (input: InputChoice) => void;
  setImage: (image: Extract<SimInput, { kind: "image" }>, name: string) => void;
};

let worker: Worker | null = null;
let frame: number | null = null;
let lastTick = 0;

function stopWorker(): void {
  worker?.terminate();
  worker = null;
}

export const useSimStore = create<SimState>((set, get) => {
  /** Moves the clock and publishes the step when it changes. */
  const setTime = (time: number) => {
    const { plan } = get();
    if (!plan) return;
    simClock.time = Math.max(0, Math.min(plan.total, time));
    const state = stateAt(plan, simClock.time);
    if (state.step !== get().step || state.ended !== get().ended) {
      set({ step: state.step, ended: state.ended });
    }
  };

  /** True when every layer of a step has been computed. */
  const stepReady = (index: number): boolean => {
    const { plan, views } = get();
    return plan?.steps[index]?.memberIds.every((id) => views[id] !== undefined) ?? true;
  };

  const tick = (now: number) => {
    frame = null;
    const { plan, playing, speed, loop, status } = get();
    if (!plan || !playing) return;
    const delta = Math.min(0.1, (now - lastTick) / 1000) * speed;
    lastTick = now;

    let next = simClock.time + delta;
    // Never show a layer's result before it exists: hold just before its "compute" phase.
    const target = stateAt(plan, next);
    const step = plan.steps[target.step];
    let waiting = false;
    if (step && !stepReady(target.step) && status === "computing") {
      const hold = step.start + step.duration * 0.3;
      if (next > hold) {
        next = hold;
        waiting = true;
      }
    }
    if (waiting !== get().waiting) set({ waiting });

    if (next >= plan.total) {
      if (loop) setTime(0);
      else {
        setTime(plan.total);
        set({ playing: false });
        return;
      }
    } else setTime(next);
    frame = requestAnimationFrame(tick);
  };

  const startLoop = () => {
    if (frame !== null) return;
    lastTick = performance.now();
    frame = requestAnimationFrame(tick);
  };

  return {
    plan: null,
    status: "idle",
    error: null,
    views: {},
    computed: 0,
    total: 0,
    estimateSeconds: 0,
    input: "shapes",
    image: null,
    imageName: null,
    playing: false,
    waiting: false,
    speed: 1,
    loop: false,
    follow: false,
    step: 0,
    ended: false,

    run: () => {
      const { ir, activeModelId } = useModelStore.getState();
      const { input, image } = get();
      if (!ir || !activeModelId) return;
      stopWorker();
      const source: SimInput =
        input === "image" && image ? image : { kind: input === "image" ? "shapes" : input };
      set({
        status: "computing",
        error: null,
        views: {},
        computed: 0,
        total: ir.index.topoOrder[activeModelId]?.length ?? 0,
      });

      worker = new Worker(new URL("../features/sim/sim.worker.ts", import.meta.url), {
        type: "module",
      });
      const current = worker;
      current.onmessage = (event: MessageEvent<SimMessage>) => {
        if (worker !== current) return; // a newer run replaced this one
        const message = event.data;
        if (message.type === "view") {
          set({
            views: { ...get().views, [message.view.nodeId]: message.view },
            computed: message.done,
          });
        } else if (message.type === "finished") {
          set({ status: "ready", waiting: false });
          stopWorker();
        } else {
          set({ status: "error", error: message.message, playing: false, waiting: false });
          stopWorker();
        }
      };
      current.onerror = () => {
        if (worker !== current) return;
        set({
          status: "error",
          error: "The simulation could not start in this browser.",
          playing: false,
        });
        stopWorker();
      };
      const request: SimRequest = { ir, modelId: activeModelId, input: source };
      current.postMessage(request);
    },

    stop: () => {
      stopWorker();
      if (get().status === "computing") set({ status: "idle", waiting: false, playing: false });
    },

    play: () => {
      const state = get();
      if (!state.plan) return;
      if (state.status === "idle" || state.status === "error") {
        state.run();
        useUiStore.getState().setInspectorTab("tensor");
      }
      if (state.ended) setTime(0);
      set({ playing: true });
      startLoop();
    },
    pause: () => set({ playing: false, waiting: false }),
    toggle: () => (get().playing ? get().pause() : get().play()),

    seek: (time) => {
      setTime(time);
    },
    stepBy: (delta) => {
      const { plan, step, ended } = get();
      if (!plan) return;
      if (get().status === "idle") {
        get().run();
        useUiStore.getState().setInspectorTab("tensor");
      }
      set({ playing: false, waiting: false });
      // "Next" from the last step goes to the very end, so the final output is shown.
      const target = (ended ? plan.steps.length : step) + delta;
      if (target >= plan.steps.length) setTime(plan.total);
      else setTime(timeOfStep(plan, target) + (delta > 0 ? 0 : 0));
    },
    setSpeed: (speed) => set({ speed }),
    setLoop: (loop) => set({ loop }),
    setFollow: (follow) => set({ follow }),

    setInput: (input) => {
      if (input === get().input) return;
      set({ input });
      // Results belong to the input they were computed from.
      if (get().status !== "idle") get().run();
    },
    setImage: (image, name) => {
      set({ image, imageName: name, input: "image" });
      if (get().status !== "idle") get().run();
    },
  };
});

/**
 * The plan follows the model on screen. A different model (or different
 * numbers after an assumption changed) throws the old results away; opening
 * or closing a group only regroups the steps.
 */
// Start with nothing shown, so a model that was opened before this store loaded is picked up too.
let shownIr: unknown = undefined;
let shownModel: string | null | undefined = undefined;
let shownFocus: unknown = null;
function followModel(model: ReturnType<typeof useModelStore.getState>): void {
  const sim = useSimStore.getState();
  const changedModel = model.ir !== shownIr || model.activeModelId !== shownModel;
  shownIr = model.ir;
  shownModel = model.activeModelId;

  const whole =
    model.ir && model.activeModelId
      ? buildPlan(buildVisibleGraph(model.ir, model.activeModelId, new Set(model.collapsed)))
      : null;
  // With a Lens focus, only the focused layers are stepped through.
  const plan = whole && model.focus ? focusPlan(whole, model.focus.nodeIds) : whole;
  const changedFocus = model.focus !== shownFocus;
  shownFocus = model.focus;
  const samePlan =
    !changedModel &&
    sim.plan !== null &&
    plan !== null &&
    sim.plan.steps.length === plan.steps.length &&
    sim.plan.steps.every((step, index) => step.itemId === plan.steps[index]!.itemId);
  if (samePlan) return;

  if (changedModel) {
    stopWorker();
    simClock.time = 0;
    useSimStore.setState({
      plan,
      status: "idle",
      error: null,
      views: {},
      computed: 0,
      total: 0,
      playing: false,
      waiting: false,
      step: 0,
      ended: false,
      estimateSeconds:
        model.ir && model.activeModelId
          ? estimateWork(model.ir, model.activeModelId) / WORK_PER_SECOND
          : 0,
    });
    return;
  }
  if (changedFocus) {
    // Entering or leaving a focus starts its pass from the beginning; computed results are kept.
    simClock.time = 0;
    useSimStore.setState({ plan, step: 0, ended: false, playing: false, waiting: false });
    return;
  }
  // Same model, regrouped: stay at the same point of the pass.
  const fraction = sim.plan && sim.plan.total > 0 ? simClock.time / sim.plan.total : 0;
  simClock.time = plan ? fraction * plan.total : 0;
  const state = plan ? stateAt(plan, simClock.time) : null;
  useSimStore.setState({ plan, step: state?.step ?? 0, ended: state?.ended ?? false });
}
useModelStore.subscribe(followModel);
followModel(useModelStore.getState());
