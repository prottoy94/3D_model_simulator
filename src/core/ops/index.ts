import { basicOps } from "./basic";
import { convOps } from "./conv";
import { extraOps } from "./extra";
import type { OpKind } from "./kinds";
import { linearOps } from "./linear";
import { poolOps } from "./pool";
import { structureOps } from "./structure";
import type { AnyOpSpec } from "./types";

export * from "./kinds";
export * from "./types";

const entries = [...basicOps, ...convOps, ...poolOps, ...linearOps, ...structureOps, ...extraOps];

/** The op catalog: the single source of truth for what each operation is. */
export const OP_CATALOG: Partial<Record<OpKind, AnyOpSpec>> = Object.fromEntries(
  entries.map((spec) => [spec.op, spec]),
);

export function getOp(op: OpKind): AnyOpSpec | undefined {
  return OP_CATALOG[op];
}

export const CATALOG_OPS: OpKind[] = entries.map((spec) => spec.op);

const lastSegment = (name: string) => name.slice(name.lastIndexOf(".") + 1);

function aliasIndex(source: "keras" | "pytorch"): Map<string, OpKind> {
  const index = new Map<string, OpKind>();
  for (const spec of entries) {
    for (const alias of spec.aliases[source] ?? []) {
      const key = lastSegment(alias);
      if (!index.has(key)) index.set(key, spec.op);
    }
  }
  return index;
}

const KERAS_ALIASES = aliasIndex("keras");
const PYTORCH_ALIASES = aliasIndex("pytorch");

/**
 * Finds the op for a layer class as written in code or in a summary table:
 * "layers.Conv2D", "tf.keras.layers.Conv2D", "nn.Conv2d" or just "Conv2D".
 */
export function opForLayerName(name: string, framework?: "keras" | "pytorch"): OpKind | undefined {
  const key = lastSegment(name.trim());
  if (framework === "keras") return KERAS_ALIASES.get(key);
  if (framework === "pytorch") return PYTORCH_ALIASES.get(key);
  return KERAS_ALIASES.get(key) ?? PYTORCH_ALIASES.get(key);
}

/** All layer class names (last segment) known for a framework, for prefix matching. */
export function layerNames(framework: "keras" | "pytorch"): Array<[string, OpKind]> {
  return [...(framework === "keras" ? KERAS_ALIASES : PYTORCH_ALIASES)];
}

/** Maps a fused or standalone activation name to its op, when it has one. */
export function opForActivation(name: string): OpKind | undefined {
  const key = name.toLowerCase();
  return (
    ["relu", "gelu", "sigmoid", "softmax", "tanh", "silu", "leaky_relu", "hardswish"] as const
  ).find((op) => op === key);
}
