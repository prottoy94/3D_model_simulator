/**
 * Wire IR v1.0 (doc 03 §1): the transport format produced by an LLM, the
 * fast-path parser or Forge. It is flat (no recursion) and every object is
 * strict, so it also works as a structured-output schema.
 */
import { z } from "zod";
import { ALGORITHM_KINDS, OP_KINDS } from "../ops/kinds";

export const IR_VERSION = "1.0";

export const codeRefSchema = z.strictObject({
  /** Notebook cell index (0-based, as in the file). */
  cell: z.number().int(),
  /** 1-based, inclusive, within the cell. */
  startLine: z.number().int(),
  endLine: z.number().int(),
  role: z.enum(["declare", "apply", "definition", "call", "loop", "config"]),
});

export const paramValueSchema = z.union([z.string(), z.number(), z.boolean(), z.array(z.number())]);

export const paramEntrySchema = z.strictObject({ key: z.string(), value: paramValueSchema });

export const wireSymbolSchema = z.strictObject({
  name: z.string(),
  /** JSON-encoded literal: "224", "[224,224,3]", "\"relu\"". */
  value: z.string(),
  provenance: z.enum(["literal", "computed", "from-output", "assumed"]),
  ref: codeRefSchema.nullable(),
  note: z.string(),
});

export const wireModelSchema = z.strictObject({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  rootGroupId: z.string(),
  inputNodeIds: z.array(z.string()),
  outputNodeIds: z.array(z.string()),
  isPrimary: z.boolean(),
  variantOf: z.string().nullable(),
});

export const wireGroupSchema = z.strictObject({
  id: z.string(),
  /** null only for a model's root group. */
  parentId: z.string().nullable(),
  modelId: z.string(),
  label: z.string(),
  kind: z.enum(["model", "stage", "block", "repeat", "module", "backbone", "head", "branch"]),
  repeatCount: z.number().int().nullable(),
  repeatSymbol: z.string().nullable(),
  zooKey: z.string().nullable(),
  zooArgs: z.array(paramEntrySchema),
  refs: z.array(codeRefSchema),
  description: z.string(),
});

export const wireNodeSchema = z.strictObject({
  id: z.string(),
  groupId: z.string(),
  op: z.enum(OP_KINDS),
  /** op "custom": the user's class name. For other ops: the user's class, when it wraps a known op. */
  customOpName: z.string().nullable(),
  label: z.string(),
  params: z.array(paramEntrySchema),
  trainable: z.enum(["yes", "no", "partial", "n/a"]),
  /** The author's belief, native layout, "B" = batch. Only a cross-check. */
  claimedOutputShape: z.array(z.union([z.number(), z.string()])).nullable(),
  refs: z.array(codeRefSchema),
  explanation: z.string(),
  confidence: z.enum(["high", "medium", "low"]),
});

export const wireEdgeSchema = z.strictObject({
  id: z.string(),
  from: z.string(),
  to: z.string(),
  /** Input slot, in order (concat order; attention: query 0, value 1, key 2). */
  toPort: z.number().int(),
  kind: z.enum(["data", "residual", "skip", "broadcast", "control"]),
});

export const wirePipelineSchema = z.strictObject({
  id: z.string(),
  label: z.string(),
  stages: z.array(
    z.strictObject({
      id: z.string(),
      label: z.string(),
      kind: z.enum(["preprocess", "algorithm", "model", "postprocess", "loss"]),
      algorithm: z.enum(ALGORITHM_KINDS).nullable(),
      modelId: z.string().nullable(),
      params: z.array(paramEntrySchema),
      refs: z.array(codeRefSchema),
      description: z.string(),
    }),
  ),
});

export const wireTrainingSchema = z.strictObject({
  optimizer: z.string().nullable(),
  learningRate: z.number().nullable(),
  loss: z.string().nullable(),
  metrics: z.array(z.string()),
  epochs: z.number().nullable(),
  batchSize: z.number().nullable(),
  phases: z.array(
    z.strictObject({
      name: z.string(),
      frozenGroupIds: z.array(z.string()),
      unfreezeRule: z.string().nullable(),
      learningRate: z.number().nullable(),
      ref: codeRefSchema.nullable(),
    }),
  ),
});

export const wireIRSchema = z.strictObject({
  irVersion: z.literal(IR_VERSION),
  scope: z.enum(["notebook", "snippet", "forge"]),
  title: z.string(),
  summary: z.string(),
  framework: z.enum([
    "keras",
    "pytorch",
    "tensorflow",
    "jax",
    "sklearn",
    "huggingface",
    "ultralytics",
    "mixed",
    "other",
  ]),
  domain: z.enum(["cv", "nlp", "tabular", "audio", "timeseries", "multimodal", "other"]),
  task: z.enum([
    "classification",
    "detection",
    "segmentation",
    "regression",
    "generation",
    "sequence-labeling",
    "clustering",
    "embedding",
    "multi-task",
    "other",
  ]),
  symbols: z.array(wireSymbolSchema),
  models: z.array(wireModelSchema),
  groups: z.array(wireGroupSchema),
  nodes: z.array(wireNodeSchema),
  edges: z.array(wireEdgeSchema),
  pipelines: z.array(wirePipelineSchema),
  training: wireTrainingSchema.nullable(),
  warnings: z.array(z.string()),
});

export type CodeRef = z.infer<typeof codeRefSchema>;
export type ParamValue = z.infer<typeof paramValueSchema>;
export type ParamEntry = z.infer<typeof paramEntrySchema>;
export type WireSymbol = z.infer<typeof wireSymbolSchema>;
export type WireModel = z.infer<typeof wireModelSchema>;
export type WireGroup = z.infer<typeof wireGroupSchema>;
export type WireNode = z.infer<typeof wireNodeSchema>;
export type WireEdge = z.infer<typeof wireEdgeSchema>;
export type WirePipeline = z.infer<typeof wirePipelineSchema>;
export type WireTraining = z.infer<typeof wireTrainingSchema>;
export type WireIR = z.infer<typeof wireIRSchema>;

/**
 * Param key any node may carry: the layer is wrapped in Keras `TimeDistributed`,
 * so it runs once per item along axis 1 (per RoI in Fast R-CNN).
 */
export const TIME_DISTRIBUTED_KEY = "timeDistributed";
