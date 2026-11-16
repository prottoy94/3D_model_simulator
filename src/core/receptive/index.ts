/**
 * Receptive fields: how much of the input image one output position of a layer can see.
 *
 * Computed from the layer settings alone (kernel, stride, dilation), walking the
 * model in order: `size += (kernel − 1) × jump`, then `jump ×= stride`, where
 * `jump` is the distance in input pixels between two neighbouring output positions.
 * This is the theoretical field. The effective one (where most of the influence
 * comes from) is smaller and depends on the trained weights, which we do not have.
 */
import type { RuntimeIR, RuntimeNode } from "../ir";
import type { OpKind } from "../ops";
import { image } from "../shapes/tensor";

export type ReceptiveField = {
  /** Height and width of the field in input pixels. Not clipped to the image. */
  size: [number, number];
  /** Input pixels between two neighbouring output positions. */
  jump: [number, number];
  /** True once a layer has mixed every position (global pooling, flatten, attention). */
  global: boolean;
  /** Height and width of the input image. */
  input: [number, number];
};

/** Layers after which one value depends on the whole input. */
const GLOBAL_OPS: ReadonlySet<OpKind> = new Set<OpKind>([
  "global_avg_pool",
  "global_max_pool",
  "adaptive_avg_pool2d",
  "flatten",
  "multi_head_attention",
  "channel_attention",
  "squeeze_excite",
  "roi_pool",
  "roi_align",
  "reduce_mean",
  "reduce_max",
  "rnn",
  "lstm",
  "gru",
  "box_decode",
  "nms",
]);

/** Layers whose inside we do not know, so nothing after them can be computed. */
const OPAQUE_OPS: ReadonlySet<OpKind> = new Set<OpKind>([
  "backbone",
  "custom",
  "estimator",
  "conv_transpose2d",
]);

const WINDOW_OPS: ReadonlySet<OpKind> = new Set<OpKind>([
  "conv2d",
  "depthwise_conv2d",
  "separable_conv2d",
  "maxpool2d",
  "avgpool2d",
  "spatial_attention",
  "patchify",
  "patch_embed",
]);

function pairOf(value: unknown): [number, number] | null {
  if (typeof value === "number" && Number.isFinite(value)) return [value, value];
  if (
    Array.isArray(value) &&
    value.length === 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number"
  ) {
    return [value[0], value[1]];
  }
  return null;
}

/** The sliding window of a layer: kernel (with dilation applied) and stride. */
function windowOf(node: RuntimeNode): { kernel: [number, number]; stride: [number, number] } {
  const p = node.params;
  const one: [number, number] = [1, 1];
  if (node.op === "maxpool2d" || node.op === "avgpool2d") {
    const kernel = pairOf(p.pool) ?? one;
    return { kernel, stride: pairOf(p.stride) ?? kernel };
  }
  if (node.op === "patchify" || node.op === "patch_embed") {
    // A patch layer on tokens (no `patch`) is a plain per-token projection.
    const kernel = pairOf(p.patch) ?? one;
    return { kernel, stride: kernel };
  }
  const kernel = pairOf(p.kernel) ?? one;
  const dilation = pairOf(p.dilation) ?? one;
  return {
    kernel: [dilation[0] * (kernel[0] - 1) + 1, dilation[1] * (kernel[1] - 1) + 1],
    stride: node.op === "spatial_attention" ? one : (pairOf(p.stride) ?? one),
  };
}

/**
 * The receptive field of every node of one model. A node maps to null when the
 * field cannot be computed: the model's input is not an image, or the node comes
 * after a layer whose inside is unknown (a pretrained backbone, a custom layer).
 */
export function receptiveFields(
  ir: RuntimeIR,
  modelId: string,
): Record<string, ReceptiveField | null> {
  const fields: Record<string, ReceptiveField | null> = {};
  for (const id of ir.index.topoOrder[modelId] ?? []) {
    const node = ir.nodes[id];
    if (!node) continue;

    if (node.op === "input") {
      try {
        const { h, w } = image(node.outShape ?? { dims: [], layout: "any" });
        fields[id] = { size: [1, 1], jump: [1, 1], global: false, input: [h, w] };
      } catch {
        fields[id] = null;
      }
      continue;
    }

    const inputs = (ir.index.inEdges[id] ?? []).map((edge) => fields[edge.from] ?? null);
    const known = inputs.filter((field): field is ReceptiveField => field !== null);
    if (known.length === 0 || known.length < inputs.length || OPAQUE_OPS.has(node.op)) {
      fields[id] = null;
      continue;
    }

    // Several inputs (add, concat): the output sees whatever any of them sees.
    const merged: ReceptiveField = {
      size: [Math.max(...known.map((f) => f.size[0])), Math.max(...known.map((f) => f.size[1]))],
      jump: [Math.max(...known.map((f) => f.jump[0])), Math.max(...known.map((f) => f.jump[1]))],
      global: known.some((f) => f.global),
      input: known[0]!.input,
    };

    if (GLOBAL_OPS.has(node.op)) {
      merged.global = true;
    } else if (WINDOW_OPS.has(node.op) && !merged.global) {
      const { kernel, stride } = windowOf(node);
      merged.size = [
        merged.size[0] + (kernel[0] - 1) * merged.jump[0],
        merged.size[1] + (kernel[1] - 1) * merged.jump[1],
      ];
      merged.jump = [merged.jump[0] * stride[0], merged.jump[1] * stride[1]];
    } else if (node.op === "upsample" && !merged.global) {
      const scale = pairOf(node.params.scale);
      // With an exact output size the factor comes from the shapes instead.
      if (scale) merged.jump = [merged.jump[0] / scale[0], merged.jump[1] / scale[1]];
      else {
        fields[id] = null;
        continue;
      }
    }
    fields[id] = merged;
  }
  return fields;
}

/** The part of the input image the field covers, as a fraction per axis (at most 1). */
export function coverage(field: ReceptiveField): [number, number] {
  if (field.global) return [1, 1];
  return [Math.min(1, field.size[0] / field.input[0]), Math.min(1, field.size[1] / field.input[1])];
}

/** "67 × 67 px" or "whole image". */
export function describeField(field: ReceptiveField): string {
  if (field.global) return "whole image";
  const [h, w] = field.size;
  if (h >= field.input[0] && w >= field.input[1]) return `whole image (${h} × ${w} px)`;
  return `${h} × ${w} px`;
}
