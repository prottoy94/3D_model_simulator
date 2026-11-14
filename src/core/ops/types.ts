import { z } from "zod";
import type { Framework, TensorShape } from "../shapes/tensor";
import type { OpFamily, OpKind } from "./kinds";

export type ParamCount = { trainable: number; nonTrainable: number };

export type ShapeCtx = { framework: Framework };

export type GlyphKind =
  | "slab" // input image / tensor
  | "volume" // feature-map producing ops (conv)
  | "plate" // thin per-element ops (norm, activation)
  | "funnel" // pooling
  | "neuron-column" // dense
  | "matrix" // embeddings, projections
  | "ring" // attention
  | "ribbon" // reshape / flatten
  | "merge" // add, concat, multiply
  | "ghost" // dropout
  | "capsule" // backbone / composite
  | "box"; // custom, detection

export type VisualizerId =
  | "none"
  | "conv-sweep"
  | "pool-window"
  | "global-pool"
  | "activation-curve"
  | "norm-histogram"
  | "dense-matrix"
  | "flatten-unroll"
  | "dropout-mask"
  | "merge-streams"
  | "softmax-bars"
  | "patchify-grid"
  | "pos-embed"
  | "attention-heads"
  | "roi-grid"
  | "embedding-lookup";

/**
 * How a layer constructor call maps onto canonical params, for the fast-path
 * parser. `positional` lists the constructor's argument names in order; `keys`
 * maps argument names to canonical param keys. Arguments not in `keys` are ignored.
 */
export type CallSignature = { positional: string[]; keys: Record<string, string> };

export type OpIO = { in: TensorShape[]; out: TensorShape };

/** One complete catalog entry (golden rule 7). */
export interface OpSpec<P> {
  op: OpKind;
  family: OpFamily;
  title: string;
  aliases: { keras?: string[]; pytorch?: string[]; functional?: string[] };
  params: z.ZodType<P>;
  /** Framework defaults merged under the given params before parsing. */
  defaults: { keras: Record<string, unknown>; pytorch: Record<string, unknown> };
  signatures: { keras?: CallSignature; pytorch?: CallSignature };
  /** Number of inputs. */
  arity: { min: number; max: number | null };
  /** True when shape and counts are declared by the IR instead of computed. */
  declared?: boolean;
  /** Throws ShapeError when the inputs do not fit. */
  inferShape(inputs: TensorShape[], p: P, ctx: ShapeCtx): TensorShape;
  countParams(inputs: TensorShape[], p: P, ctx: ShapeCtx): ParamCount;
  /** Multiply-adds × 2, per sample. */
  flops(inputs: TensorShape[], out: TensorShape, p: P): number;
  glyph: GlyphKind;
  visualizer: VisualizerId;
  explain(p: P, io: OpIO): string;
  codegen: { pytorch(p: P, io: OpIO): string; keras(p: P, io: OpIO): string };
}

/** A catalog entry with its param type erased, for lookups by op name. */
export type AnyOpSpec = OpSpec<unknown>;

export function defineOp<P>(spec: OpSpec<P>): AnyOpSpec {
  // Safe: the engine only ever calls a spec with params parsed by that spec's own schema.
  return spec as unknown as AnyOpSpec;
}

export const NO_PARAMS: ParamCount = { trainable: 0, nonTrainable: 0 };

// ── Shared param schemas ──────────────────────────────────────────────

export const posInt = z.number().int().positive();

/** `3` or `[3, 3]` → `[3, 3]`. */
export const pair = z.preprocess(
  (value) => (typeof value === "number" ? [value, value] : value),
  z.tuple([posInt, posInt]),
);

const nonNegPair = z.tuple([z.number().int().nonnegative(), z.number().int().nonnegative()]);

/** "valid" / "same" (any case), or explicit padding per side as `1` or `[1, 1]`. */
export const padding = z.preprocess(
  (value) => {
    if (typeof value === "string") return value.toLowerCase();
    if (typeof value === "number") return [value, value];
    return value;
  },
  z.union([z.enum(["valid", "same"]), nonNegPair]),
);

export type Padding = z.infer<typeof padding>;

/** Fused activation name. `None` / missing means linear. */
export const activation = z.preprocess(
  (value) => (value === null || value === undefined || value === "" ? "linear" : value),
  z.string(),
);

// ── Code generation helpers ───────────────────────────────────────────

export const tuple = (values: readonly number[]) => `(${values.join(", ")})`;

export function kerasArgs(parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(", ");
}

export const kerasActivation = (name: string) =>
  name === "linear" ? null : `activation="${name}"`;

export const times = (values: readonly number[]) => values.join("×");
