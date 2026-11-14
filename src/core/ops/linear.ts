/** Dense, embeddings, patch ops and attention. */
import { z } from "zod";
import {
  BATCH,
  image,
  isImage,
  known,
  lastDim,
  positions,
  ShapeError,
  shapeText,
  withLastDim,
  type TensorShape,
} from "../shapes/tensor";
import {
  activation,
  defineOp,
  kerasActivation,
  kerasArgs,
  NO_PARAMS,
  pair,
  posInt,
  times,
  tuple,
} from "./types";

const denseParams = z.object({
  units: posInt,
  bias: z.boolean().default(true),
  activation: activation.default("linear"),
});

type DenseParams = z.infer<typeof denseParams>;

export const dense = defineOp<DenseParams>({
  op: "dense",
  family: "linear",
  title: "Dense (fully connected)",
  aliases: { keras: ["Dense"], pytorch: ["nn.Linear"], functional: ["F.linear"] },
  params: denseParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {
    keras: {
      positional: ["units", "activation", "use_bias"],
      keys: { units: "units", activation: "activation", use_bias: "bias" },
    },
    pytorch: {
      positional: ["in_features", "out_features", "bias"],
      keys: { out_features: "units", bias: "bias" },
    },
  },
  arity: { min: 1, max: 1 },
  // Dense works on the last axis, so it also accepts (B, tokens, features).
  inferShape: ([shape], p) => {
    lastDim(shape!, "input feature size");
    return withLastDim(shape!, p.units);
  },
  countParams: ([shape], p) => ({
    trainable: lastDim(shape!) * p.units + (p.bias ? p.units : 0),
    nonTrainable: 0,
  }),
  flops: ([shape], _out, p) => 2 * lastDim(shape!) * p.units * positions(shape!),
  glyph: "neuron-column",
  visualizer: "dense-matrix",
  explain: (p, io) => {
    const inSize = lastDim(io.in[0]!);
    const perToken =
      io.in[0]!.dims.length > 2 ? " This happens separately for every position." : "";
    const act = p.activation === "linear" ? "" : `, then ${p.activation}`;
    return `Each of the ${p.units} outputs is a weighted sum of all ${inSize} inputs${p.bias ? " plus a bias" : ""}${act}. That is a ${inSize}×${p.units} weight matrix.${perToken}`;
  },
  codegen: {
    keras: (p) =>
      `layers.Dense(${kerasArgs([String(p.units), kerasActivation(p.activation), !p.bias && "use_bias=False"])})`,
    pytorch: (p, io) =>
      `nn.Linear(${lastDim(io.in[0]!)}, ${p.units}${p.bias ? "" : ", bias=False"})`,
  },
});

const embeddingParams = z.object({ vocab: posInt, dim: posInt });

type EmbeddingParams = z.infer<typeof embeddingParams>;

export const embedding = defineOp<EmbeddingParams>({
  op: "embedding",
  family: "linear",
  title: "Embedding",
  aliases: { keras: ["Embedding"], pytorch: ["nn.Embedding"] },
  params: embeddingParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {
    keras: {
      positional: ["input_dim", "output_dim"],
      keys: { input_dim: "vocab", output_dim: "dim" },
    },
    pytorch: {
      positional: ["num_embeddings", "embedding_dim"],
      keys: { num_embeddings: "vocab", embedding_dim: "dim" },
    },
  },
  arity: { min: 1, max: 1 },
  inferShape: ([shape], p) => ({
    dims: [...shape!.dims, p.dim],
    layout: shape!.dims.length === 2 ? "NLC" : "any",
  }),
  countParams: (_inputs, p) => ({ trainable: p.vocab * p.dim, nonTrainable: 0 }),
  flops: () => 0,
  glyph: "matrix",
  visualizer: "embedding-lookup",
  explain: (p) =>
    `A lookup table with ${p.vocab} rows of ${p.dim} numbers. Each integer id picks its row, so ids become learned vectors.`,
  codegen: {
    keras: (p) => `layers.Embedding(${p.vocab}, ${p.dim})`,
    pytorch: (p) => `nn.Embedding(${p.vocab}, ${p.dim})`,
  },
});

const patchifyParams = z.object({ patch: pair });

type PatchifyParams = z.infer<typeof patchifyParams>;

export const patchify = defineOp<PatchifyParams>({
  op: "patchify",
  family: "linear",
  title: "Split into patches",
  aliases: { functional: ["tf.image.extract_patches", "F.unfold"] },
  params: patchifyParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 1, max: 1 },
  inferShape: ([shape], p) => {
    const { h, w, c } = image(shape!);
    const tokens = Math.floor(h / p.patch[0]) * Math.floor(w / p.patch[1]);
    if (tokens < 1) throw new ShapeError("the patch is larger than the image");
    return { dims: [shape!.dims[0] ?? BATCH, tokens, p.patch[0] * p.patch[1] * c], layout: "NLC" };
  },
  countParams: () => NO_PARAMS,
  flops: () => 0,
  glyph: "ribbon",
  visualizer: "patchify-grid",
  explain: (p, io) => {
    const { h, w, c } = image(io.in[0]!);
    const tokens = known(io.out.dims[1], "tokens");
    return `The ${h}×${w} image is cut into a ${Math.floor(h / p.patch[0])}×${Math.floor(w / p.patch[1])} grid of ${times(p.patch)} patches. Each patch is flattened into ${p.patch[0] * p.patch[1] * c} numbers, giving ${tokens} tokens. No weights.`;
  },
  codegen: {
    keras: (p) => `Patches(${p.patch[0]})  # tf.image.extract_patches + reshape`,
    pytorch: (p) => `nn.Unfold(kernel_size=${tuple(p.patch)}, stride=${tuple(p.patch)})`,
  },
});

const patchEmbedParams = z.object({
  dim: posInt,
  /** Needed when the input is still an image (conv-style patch embedding). */
  patch: pair.optional(),
  bias: z.boolean().default(true),
});

type PatchEmbedParams = z.infer<typeof patchEmbedParams>;

function patchEmbedInput(
  shape: TensorShape,
  p: PatchEmbedParams,
): { tokens: number | null; inSize: number } {
  if (isImage(shape)) {
    if (!p.patch) throw new ShapeError("patch size is needed to embed an image");
    const { h, w, c } = image(shape);
    return {
      tokens: Math.floor(h / p.patch[0]) * Math.floor(w / p.patch[1]),
      inSize: p.patch[0] * p.patch[1] * c,
    };
  }
  if (shape.dims.length !== 3)
    throw new ShapeError(`expected patches (B, tokens, size), got ${shapeText(shape)}`);
  return { tokens: null, inSize: lastDim(shape) };
}

export const patchEmbed = defineOp<PatchEmbedParams>({
  op: "patch_embed",
  family: "linear",
  title: "Patch projection",
  aliases: {},
  params: patchEmbedParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 1, max: 1 },
  inferShape: ([shape], p) => {
    const { tokens } = patchEmbedInput(shape!, p);
    return tokens === null
      ? withLastDim(shape!, p.dim)
      : { dims: [shape!.dims[0] ?? BATCH, tokens, p.dim], layout: "NLC" };
  },
  countParams: ([shape], p) => {
    const { inSize } = patchEmbedInput(shape!, p);
    return { trainable: inSize * p.dim + (p.bias ? p.dim : 0), nonTrainable: 0 };
  },
  flops: ([shape], out, p) => 2 * patchEmbedInput(shape!, p).inSize * p.dim * positions(out),
  glyph: "matrix",
  visualizer: "dense-matrix",
  explain: (p, io) =>
    `Every patch (${patchEmbedInput(io.in[0]!, p).inSize} numbers) is multiplied by the same learned matrix to give a ${p.dim}-number token. All patches share these weights.`,
  codegen: {
    keras: (p) => `layers.Dense(${p.dim})`,
    pytorch: (p, io) =>
      p.patch && isImage(io.in[0]!)
        ? `nn.Conv2d(${image(io.in[0]!).c}, ${p.dim}, kernel_size=${tuple(p.patch)}, stride=${tuple(p.patch)})`
        : `nn.Linear(${lastDim(io.in[0]!)}, ${p.dim})`,
  },
});

const posEmbedParams = z.object({
  /** Both default to the input's token count and width. */
  length: posInt.optional(),
  dim: posInt.optional(),
  learned: z.boolean().default(true),
});

type PosEmbedParams = z.infer<typeof posEmbedParams>;

function posEmbedSize(shape: TensorShape, p: PosEmbedParams): { length: number; dim: number } {
  if (shape.dims.length !== 3)
    throw new ShapeError(`expected tokens (B, tokens, size), got ${shapeText(shape)}`);
  const tokens = known(shape.dims[1], "token count");
  const width = lastDim(shape);
  const length = p.length ?? tokens;
  const dim = p.dim ?? width;
  if (length !== tokens || dim !== width) {
    throw new ShapeError(
      `position table is ${length}×${dim} but the tokens are ${tokens}×${width}`,
    );
  }
  return { length, dim };
}

export const posEmbed = defineOp<PosEmbedParams>({
  op: "pos_embed",
  family: "linear",
  title: "Position embedding",
  aliases: {},
  params: posEmbedParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {},
  arity: { min: 1, max: 1 },
  inferShape: ([shape], p) => {
    posEmbedSize(shape!, p);
    return shape!;
  },
  countParams: ([shape], p) => {
    const { length, dim } = posEmbedSize(shape!, p);
    return { trainable: p.learned ? length * dim : 0, nonTrainable: 0 };
  },
  flops: ([shape]) => positions(shape!) * lastDim(shape!),
  glyph: "matrix",
  visualizer: "pos-embed",
  explain: (p, io) => {
    const { length, dim } = posEmbedSize(io.in[0]!, p);
    return `A ${p.learned ? "learned" : "fixed"} vector of ${dim} numbers is added to each of the ${length} tokens, one vector per position. Without it the model could not tell where a patch came from.`;
  },
  codegen: {
    keras: (p, io) => {
      const { length, dim } = posEmbedSize(io.in[0]!, p);
      return `layers.Embedding(input_dim=${length}, output_dim=${dim})  # added to the tokens`;
    },
    pytorch: (p, io) => {
      const { length, dim } = posEmbedSize(io.in[0]!, p);
      return `nn.Parameter(torch.zeros(1, ${length}, ${dim}))  # added to the tokens`;
    },
  },
});

const mhaParams = z.object({
  numHeads: posInt,
  /** Keras: size of each head's query/key space (per head, not total). */
  keyDim: posInt.optional(),
  valueDim: posInt.optional(),
  outputDim: posInt.optional(),
  dropout: z.number().min(0).max(1).default(0),
  bias: z.boolean().default(true),
  selfAttention: z.boolean().default(true),
});

type MhaParams = z.infer<typeof mhaParams>;

/** Inputs are ordered query, value, key (Keras call order). Missing ones repeat the query. */
function mhaDims(inputs: TensorShape[], p: MhaParams) {
  const query = inputs[0];
  if (!query || query.dims.length < 3)
    throw new ShapeError("attention needs a sequence (B, tokens, size)");
  const value = inputs[1] ?? query;
  const key = inputs[2] ?? value;
  const dq = lastDim(query);
  return { dq, dv: lastDim(value), dk: lastDim(key), out: p.outputDim ?? dq };
}

export const multiHeadAttention = defineOp<MhaParams>({
  op: "multi_head_attention",
  family: "attention",
  title: "Multi-Head Attention",
  aliases: { keras: ["MultiHeadAttention"], pytorch: ["nn.MultiheadAttention"] },
  params: mhaParams,
  defaults: { keras: {}, pytorch: {} },
  signatures: {
    keras: {
      positional: ["num_heads", "key_dim", "value_dim", "dropout", "use_bias"],
      keys: {
        num_heads: "numHeads",
        key_dim: "keyDim",
        value_dim: "valueDim",
        dropout: "dropout",
        use_bias: "bias",
      },
    },
    pytorch: {
      positional: ["embed_dim", "num_heads", "dropout", "bias"],
      keys: { num_heads: "numHeads", dropout: "dropout", bias: "bias" },
    },
  },
  arity: { min: 1, max: 3 },
  inferShape: (inputs, p) => withLastDim(inputs[0]!, mhaDims(inputs, p).out),
  countParams: (inputs, p, ctx) => {
    const { dq, dv, dk, out } = mhaDims(inputs, p);
    const b = p.bias ? 1 : 0;
    if (ctx.framework === "pytorch") {
      // nn.MultiheadAttention: one 3E×E input projection and one E×E output projection.
      if (dq % p.numHeads !== 0)
        throw new ShapeError(`embed_dim ${dq} is not divisible by ${p.numHeads} heads`);
      return { trainable: 3 * dq * dq + b * 3 * dq + dq * dq + b * dq, nonTrainable: 0 };
    }
    if (!p.keyDim) throw new ShapeError("key_dim is required for Keras MultiHeadAttention");
    const k = p.numHeads * p.keyDim;
    const v = p.numHeads * (p.valueDim ?? p.keyDim);
    return {
      trainable: dq * k + b * k + (dk * k + b * k) + (dv * v + b * v) + (v * out + b * out),
      nonTrainable: 0,
    };
  },
  flops: (inputs, _out, p) => {
    const { dq } = mhaDims(inputs, p);
    const tokens = positions(inputs[0]!);
    const width = p.numHeads * (p.keyDim ?? dq / p.numHeads);
    // Four projections per token, plus the token×token score and mixing steps.
    return 2 * tokens * 4 * dq * width + 4 * tokens * tokens * width;
  },
  glyph: "ring",
  visualizer: "attention-heads",
  explain: (p, io) => {
    const tokens = positions(io.in[0]!);
    const width = p.keyDim
      ? ` Each head works in ${p.keyDim} dimensions (key_dim is per head), so the attention space is ${p.numHeads * p.keyDim} wide.`
      : "";
    return `Each of the ${tokens} tokens looks at every other token and mixes in the ones that matter to it, using ${p.numHeads} heads in parallel.${width}`;
  },
  codegen: {
    keras: (p) =>
      `layers.MultiHeadAttention(${kerasArgs([
        `num_heads=${p.numHeads}`,
        p.keyDim !== undefined && `key_dim=${p.keyDim}`,
        p.dropout > 0 && `dropout=${p.dropout}`,
      ])})`,
    pytorch: (p, io) =>
      `nn.MultiheadAttention(${lastDim(io.in[0]!)}, ${p.numHeads}${p.dropout > 0 ? `, dropout=${p.dropout}` : ""}, batch_first=True)`,
  },
});

export const linearOps = [dense, embedding, patchify, patchEmbed, posEmbed, multiHeadAttention];
