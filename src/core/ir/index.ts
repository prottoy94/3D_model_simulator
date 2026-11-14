export { buildRuntimeIR, modelTotals, type BuildResult } from "./build";
export { ref, WireBuilder } from "./builder";
export {
  engineFramework,
  normalizeWire,
  type NormalizedIR,
  type NormalizeResult,
} from "./normalize";
export { applyRepairPatch, repairPatchSchema, type PatchResult, type RepairPatch } from "./patch";
export type * from "./runtime";
export * from "./wire";
