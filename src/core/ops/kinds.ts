/**
 * Every operation name the IR may use (doc 03 §3). The wire schema's `op` enum
 * is built from this list. An op only has shapes and counts once it also has a
 * catalog entry (see ./index.ts); the others are shown as "unverified".
 */
export const OP_KINDS = [
  // io
  "input",
  "output",
  // conv
  "conv1d",
  "conv2d",
  "conv3d",
  "depthwise_conv2d",
  "separable_conv2d",
  "conv_transpose2d",
  // pool
  "maxpool1d",
  "maxpool2d",
  "avgpool2d",
  "global_avg_pool",
  "global_max_pool",
  "adaptive_avg_pool2d",
  // norm
  "batchnorm",
  "layernorm",
  "groupnorm",
  "instancenorm",
  // activation
  "relu",
  "leaky_relu",
  "gelu",
  "silu",
  "sigmoid",
  "tanh",
  "softmax",
  "hardswish",
  // linear
  "dense",
  "embedding",
  "patchify",
  "patch_embed",
  "pos_embed",
  "cls_token",
  // attention
  "multi_head_attention",
  "channel_attention",
  "spatial_attention",
  "squeeze_excite",
  // recurrent
  "rnn",
  "lstm",
  "gru",
  "bidirectional",
  // reshape
  "flatten",
  "reshape",
  "permute",
  "slice",
  "squeeze",
  "unsqueeze",
  "upsample",
  // merge
  "add",
  "concat",
  "multiply",
  "matmul",
  "reduce_mean",
  "reduce_max",
  // regularization
  "dropout",
  "drop_path",
  // detection
  "anchor_generator",
  "roi_pool",
  "roi_align",
  "box_decode",
  "nms",
  // composite, classical, custom
  "backbone",
  "estimator",
  "custom",
] as const;

export type OpKind = (typeof OP_KINDS)[number];

export const OP_FAMILIES = [
  "io",
  "conv",
  "pool",
  "norm",
  "activation",
  "linear",
  "attention",
  "recurrent",
  "reshape",
  "merge",
  "regularization",
  "detection",
  "composite",
  "classical",
  "custom",
] as const;

export type OpFamily = (typeof OP_FAMILIES)[number];

/** Design token that colours each family (src/styles/tokens.css). */
export const FAMILY_COLOR_TOKEN: Record<OpFamily, string> = {
  io: "--op-io",
  conv: "--op-conv",
  pool: "--op-pool",
  norm: "--op-norm",
  activation: "--op-act",
  linear: "--op-linear",
  attention: "--op-attn",
  recurrent: "--op-recur",
  reshape: "--op-shape",
  merge: "--op-merge",
  regularization: "--op-reg",
  detection: "--op-det",
  composite: "--op-io",
  classical: "--op-classic",
  custom: "--op-custom",
};

/** Non-neural stages of a pipeline (doc 03 §3). */
export const ALGORITHM_KINDS = [
  "selective_search",
  "iou",
  "nms",
  "anchor_generation",
  "anchor_labeling",
  "box_encode",
  "box_decode",
  "crop_resize",
  "tokenize",
  "train_test_split",
  "gradient_descent",
  "tree_split",
  "kmeans_step",
] as const;

export type AlgorithmKind = (typeof ALGORITHM_KINDS)[number];

/** The ops Phase 2 must fully implement (roadmap), plus roi_pool for Fast R-CNN. */
export const MVP_OPS: readonly OpKind[] = [
  "input",
  "conv2d",
  "depthwise_conv2d",
  "maxpool2d",
  "avgpool2d",
  "global_avg_pool",
  "batchnorm",
  "layernorm",
  "relu",
  "gelu",
  "softmax",
  "sigmoid",
  "dense",
  "dropout",
  "flatten",
  "reshape",
  "add",
  "concat",
  "multiply",
  "patchify",
  "patch_embed",
  "pos_embed",
  "slice",
  "multi_head_attention",
  "embedding",
  "custom",
  "backbone",
  "roi_pool",
];
