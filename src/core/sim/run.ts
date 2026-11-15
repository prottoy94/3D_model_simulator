/**
 * Runs a model on one sample, layer by layer, in topological order. It is a
 * generator so the caller (a web worker) can report each layer as it finishes
 * and stop at any point.
 */
import type { RuntimeIR, RuntimeNode } from "../ir/runtime";
import { runKernel, seedFor, type KernelExtras } from "./kernels";
import { canonicalShape, illustrativeTensor, randomStream, sizeOf, type SimTensor } from "./tensor";

/** Built-in sample inputs, or a picture the user supplied (RGB, values 0..1, row-major). */
export type SimInput =
  | { kind: "shapes" | "stripes" | "noise" }
  | { kind: "image"; width: number; height: number; rgb: Float32Array };

export type NodeResult = {
  nodeId: string;
  tensor: SimTensor;
  inputs: SimTensor[];
  extras: KernelExtras;
  /** Why the values are illustrative, when they are. */
  note: string | null;
  ms: number;
};

/** Tensors above this many values are not materialized (about 160 MB of floats). */
const MAX_ELEMENTS = 40_000_000;

function sampleColor(
  input: SimInput,
  u: number,
  v: number,
  next: () => number,
): [number, number, number] {
  if (input.kind === "image") {
    const x = Math.min(input.width - 1, Math.floor(u * input.width));
    const y = Math.min(input.height - 1, Math.floor(v * input.height));
    const at = (y * input.width + x) * 3;
    return [input.rgb[at]!, input.rgb[at + 1]!, input.rgb[at + 2]!];
  }
  if (input.kind === "noise") return [next(), next(), next()];
  if (input.kind === "stripes") {
    const wave = 0.5 + 0.5 * Math.sin((u + v * 0.6) * 40);
    return [wave, 0.25 + 0.5 * wave, 1 - wave];
  }
  // "shapes": a disc, a square and a diagonal edge on a soft gradient. Easy to follow through a CNN.
  const disc = Math.hypot(u - 0.32, v - 0.36) < 0.2;
  const square = Math.abs(u - 0.7) < 0.14 && Math.abs(v - 0.66) < 0.14;
  const edge = Math.abs(u - v - 0.25) < 0.02;
  if (disc) return [1, 0.75, 0.15];
  if (square) return [0.15, 0.85, 1];
  if (edge) return [1, 1, 1];
  return [0.08 + 0.18 * u, 0.1 + 0.12 * v, 0.2 + 0.15 * (1 - u)];
}

/** The tensor fed to an input node: a picture for image inputs, sample values otherwise. */
export function makeInput(shape: number[], input: SimInput, seed: number): SimTensor {
  const data = new Float32Array(sizeOf(shape));
  const next = randomStream(seed);
  const channels = shape[shape.length - 1] ?? 1;
  if (shape.length === 3 && channels <= 4) {
    const [h, w] = shape as [number, number, number];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const color = sampleColor(input, (x + 0.5) / w, (y + 0.5) / h, next);
        const gray = (color[0] + color[1] + color[2]) / 3;
        for (let c = 0; c < channels; c++)
          data[(y * w + x) * channels + c] = channels === 3 ? color[c]! : gray;
      }
    }
  } else if (shape.length === 2 && channels === 4) {
    // Region boxes [x1, y1, x2, y2] as fractions of the image.
    for (let box = 0; box < shape[0]!; box++) {
      const x = next() * 0.5;
      const y = next() * 0.5;
      data.set([x, y, x + 0.25 + next() * 0.25, y + 0.25 + next() * 0.25], box * 4);
    }
  } else if (shape.length === 1 && channels > 4) {
    for (let i = 0; i < data.length; i++) data[i] = next();
  } else {
    // Token ids or other integer-like input.
    for (let i = 0; i < data.length; i++) data[i] = Math.floor(next() * 100);
  }
  return { shape, data, illustrative: false };
}

/** Multiply-adds the model needs for one sample, for a time estimate before running. */
export function estimateWork(ir: RuntimeIR, modelId: string): number {
  return (ir.index.topoOrder[modelId] ?? []).reduce((sum, id) => sum + ir.nodes[id]!.flops / 2, 0);
}

export function* simulate(ir: RuntimeIR, modelId: string, input: SimInput): Generator<NodeResult> {
  const order = ir.index.topoOrder[modelId] ?? [];
  const tensors = new Map<string, SimTensor>();
  // How many layers still need each tensor; it is dropped when the last one has run.
  const waiting = new Map<string, number>();
  for (const id of order) waiting.set(id, ir.index.outEdges[id]?.length ?? 0);

  for (const id of order) {
    const node: RuntimeNode = ir.nodes[id]!;
    const started = performance.now();
    const edges = ir.index.inEdges[id] ?? [];
    const inputs = edges
      .map((edge) => tensors.get(edge.from))
      .filter((item): item is SimTensor => item !== undefined);
    const extras: KernelExtras = {};
    const seed = seedFor(id);
    let note: string | null = null;
    let tensor: SimTensor | null = null;

    const expected = node.outShape ? canonicalShape(node.outShape) : null;
    if (!expected) {
      note = "Its output shape is unknown, so there is nothing to compute.";
    } else if (sizeOf(expected) > MAX_ELEMENTS) {
      note = "This tensor is too large to compute in the browser.";
    } else if (node.op === "input") {
      tensor = makeInput(expected, input, seed);
    } else if (inputs.length !== edges.length) {
      note = "An input of this layer could not be computed.";
    } else {
      try {
        tensor = runKernel(node, inputs, {
          framework: ir.engineFramework,
          expected,
          nativeInputs: node.inShapes,
          seed,
          extras,
        });
        if (!tensor) {
          note =
            node.op === "backbone"
              ? "A pretrained network is not simulated layer by layer yet. These values only show the shape of its output."
              : "This kind of layer is not simulated yet. These values only show the shape of its output.";
        } else if (tensor.data.length !== sizeOf(expected)) {
          note = "The computed size did not match the engine's shape, so the values were replaced.";
          tensor = null;
        } else {
          tensor = {
            shape: expected,
            data: tensor.data,
            illustrative: inputs.some((item) => item.illustrative),
          };
          if (tensor.illustrative)
            note = "Computed from a layer whose values are illustrative, so these are too.";
        }
      } catch {
        note =
          "The computation failed for this layer. These values only show the shape of its output.";
        tensor = null;
      }
    }

    const result =
      tensor ??
      illustrativeTensor(expected && sizeOf(expected) <= MAX_ELEMENTS ? expected : [1], seed);
    tensors.set(id, result);
    yield { nodeId: id, tensor: result, inputs, extras, note, ms: performance.now() - started };

    for (const edge of edges) {
      const left = (waiting.get(edge.from) ?? 1) - 1;
      waiting.set(edge.from, left);
      if (left <= 0) tensors.delete(edge.from);
    }
    if ((waiting.get(id) ?? 0) === 0) tensors.delete(id);
  }
}
