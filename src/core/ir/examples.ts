/**
 * Hand-written Wire IR for three reference architectures. They are the golden
 * fixtures for the shape engine (fixtures/ir/*.json is generated from them by
 * `pnpm build-ir-fixtures`) and they seed the gallery.
 *
 * Code refs point at the reference notebooks N1 (cells 14 and 26) and N4 (cell 16).
 */
import { ref, WireBuilder } from "./builder";
import type { ParamValue, WireIR } from "./wire";

const conv = (
  filters: number,
  kernel: number,
  extra: Record<string, ParamValue> = {},
): Record<string, ParamValue> => ({
  filters,
  kernel: [kernel, kernel],
  padding: "same",
  activation: "relu",
  ...extra,
});

export function alexnetIR(): WireIR {
  const cell = 14;
  const b = new WireBuilder({
    title: "AlexNet",
    summary:
      "AlexNet (2012) with BatchNorm in place of local response normalization: five convolutions, then three dense layers. Built from scratch for 8 classes.",
    framework: "keras",
  })
    .symbol("IMG_HEIGHT", 224)
    .symbol("IMG_WIDTH", 224)
    .symbol("num_classes", 8, {
      provenance: "from-output",
      note: "len(class_names); the notebook reports 8 classes",
    })
    .model("m_alexnet", "AlexNet", "AlexNet adapted with BatchNorm, trained from scratch", [
      ref(cell, 1, 37, "definition"),
      ref(cell, 40, 40, "call"),
    ])
    .node(
      "n_in",
      "input",
      { shape: [224, 224, 3] },
      { label: "Input 224×224×3", refs: [ref(cell, 12, 12)], trainable: "n/a" },
    );

  b.group("g_block1", "Conv block 1", { kind: "stage" })
    .node("n_conv1", "conv2d", conv(96, 11, { stride: [4, 4], padding: "valid" }), {
      from: ["n_in"],
      label: "Conv 1 · 96@11×11 /4",
      refs: [ref(cell, 12, 12)],
      explanation:
        "Large 11×11 filters with stride 4 shrink the image fast while detecting coarse patterns.",
    })
    .node(
      "n_bn1",
      "batchnorm",
      {},
      { from: ["n_conv1"], label: "BatchNorm 1", refs: [ref(cell, 13, 13)] },
    )
    .node(
      "n_pool1",
      "maxpool2d",
      { pool: [3, 3], stride: [2, 2] },
      { from: ["n_bn1"], label: "MaxPool 3×3 /2", refs: [ref(cell, 14, 14)], trainable: "n/a" },
    );

  b.group("g_block2", "Conv block 2", { kind: "stage" })
    .node("n_conv2", "conv2d", conv(256, 5), {
      from: ["n_pool1"],
      label: "Conv 2 · 256@5×5",
      refs: [ref(cell, 17, 17)],
    })
    .node(
      "n_bn2",
      "batchnorm",
      {},
      { from: ["n_conv2"], label: "BatchNorm 2", refs: [ref(cell, 18, 18)] },
    )
    .node(
      "n_pool2",
      "maxpool2d",
      { pool: [3, 3], stride: [2, 2] },
      { from: ["n_bn2"], label: "MaxPool 3×3 /2", refs: [ref(cell, 19, 19)], trainable: "n/a" },
    );

  b.group("g_block3", "Conv block 3", { kind: "stage" })
    .node("n_conv3", "conv2d", conv(384, 3), {
      from: ["n_pool2"],
      label: "Conv 3 · 384@3×3",
      refs: [ref(cell, 23, 23)],
    })
    .node("n_conv4", "conv2d", conv(384, 3), {
      from: ["n_conv3"],
      label: "Conv 4 · 384@3×3",
      refs: [ref(cell, 24, 24)],
    })
    .node("n_conv5", "conv2d", conv(256, 3), {
      from: ["n_conv4"],
      label: "Conv 5 · 256@3×3",
      refs: [ref(cell, 25, 25)],
    })
    .node(
      "n_pool3",
      "maxpool2d",
      { pool: [3, 3], stride: [2, 2] },
      { from: ["n_conv5"], label: "MaxPool 3×3 /2", refs: [ref(cell, 26, 26)], trainable: "n/a" },
    );

  b.group("g_head", "Classifier", { kind: "head" })
    .node(
      "n_flatten",
      "flatten",
      {},
      { from: ["n_pool3"], label: "Flatten", refs: [ref(cell, 29, 29)], trainable: "n/a" },
    )
    .node(
      "n_fc1",
      "dense",
      { units: 4096, activation: "relu" },
      { from: ["n_flatten"], label: "Dense 4096", refs: [ref(cell, 30, 30)] },
    )
    .node(
      "n_drop1",
      "dropout",
      { rate: 0.5 },
      { from: ["n_fc1"], label: "Dropout 0.5", refs: [ref(cell, 31, 31)], trainable: "n/a" },
    )
    .node(
      "n_fc2",
      "dense",
      { units: 4096, activation: "relu" },
      { from: ["n_drop1"], label: "Dense 4096", refs: [ref(cell, 32, 32)] },
    )
    .node(
      "n_drop2",
      "dropout",
      { rate: 0.5 },
      { from: ["n_fc2"], label: "Dropout 0.5", refs: [ref(cell, 33, 33)], trainable: "n/a" },
    )
    .node(
      "n_fc3",
      "dense",
      { units: "num_classes", activation: "softmax" },
      { from: ["n_drop2"], label: "Dense 8 · softmax", refs: [ref(cell, 34, 34)] },
    )
    .outputs("n_fc3");
  return b.build();
}

export function vgg16IR(): WireIR {
  const cell = 26;
  const b = new WireBuilder({
    title: "VGG16",
    summary:
      "VGG16 (configuration D): thirteen 3×3 convolutions in five blocks. This variant replaces Flatten and the two 4096-wide dense layers with global average pooling, which is why it is smaller than AlexNet.",
    framework: "keras",
  })
    .symbol("IMG_HEIGHT", 224)
    .symbol("IMG_WIDTH", 224)
    .symbol("num_classes", 8, {
      provenance: "from-output",
      note: "len(class_names); the notebook reports 8 classes",
    })
    .model("m_vgg16", "VGG16", "VGG16 with a global-average-pooling head, trained from scratch", [
      ref(cell, 1, 43, "definition"),
      ref(cell, 46, 46, "call"),
    ])
    .node(
      "n_in",
      "input",
      { shape: [224, 224, 3] },
      { label: "Input 224×224×3", refs: [ref(cell, 9, 9)], trainable: "n/a" },
    );

  // [filters, first line of the block, number of convs]
  const blocks: Array<[number, number, number]> = [
    [64, 9, 2],
    [128, 14, 2],
    [256, 19, 3],
    [512, 25, 3],
    [512, 31, 3],
  ];
  let previous = "n_in";
  blocks.forEach(([filters, firstLine, count], index) => {
    const block = index + 1;
    b.group(`g_block${block}`, `Block ${block}`, {
      kind: "stage",
      refs: [ref(cell, firstLine, firstLine + count)],
    });
    for (let i = 0; i < count; i++) {
      const id = `n_conv${block}_${i + 1}`;
      b.node(id, "conv2d", conv(filters, 3), {
        from: [previous],
        label: `Conv ${block}.${i + 1} · ${filters}@3×3`,
        refs: [ref(cell, firstLine + i, firstLine + i)],
      });
      previous = id;
    }
    const pool = `n_pool${block}`;
    b.node(
      pool,
      "maxpool2d",
      { pool: [2, 2], stride: [2, 2] },
      {
        from: [previous],
        label: "MaxPool 2×2",
        refs: [ref(cell, firstLine + count, firstLine + count)],
        trainable: "n/a",
      },
    );
    previous = pool;
  });

  b.group("g_head", "Classifier", { kind: "head" })
    .node(
      "n_gap",
      "global_avg_pool",
      {},
      {
        from: [previous],
        label: "Global average pool",
        refs: [ref(cell, 37, 37)],
        trainable: "n/a",
      },
    )
    .node(
      "n_fc1",
      "dense",
      { units: 512, activation: "relu" },
      { from: ["n_gap"], label: "Dense 512", refs: [ref(cell, 38, 38)] },
    )
    .node(
      "n_drop",
      "dropout",
      { rate: 0.5 },
      { from: ["n_fc1"], label: "Dropout 0.5", refs: [ref(cell, 39, 39)], trainable: "n/a" },
    )
    .node(
      "n_fc2",
      "dense",
      { units: "num_classes", activation: "softmax" },
      { from: ["n_drop"], label: "Dense 8 · softmax", refs: [ref(cell, 40, 40)] },
    )
    .outputs("n_fc2");
  return b.build();
}

export function miniVitIR(): WireIR {
  const cell = 16;
  const b = new WireBuilder({
    title: "Mini Vision Transformer",
    summary:
      "A small ViT built from scratch: the image is cut into 196 patches of 16×16, each becomes a 64-number token, and four Transformer blocks let the tokens exchange information before a pooled classifier head.",
    framework: "keras",
  })
    .symbol("IMG_HEIGHT", 224)
    .symbol("IMG_WIDTH", 224)
    .symbol("PATCH_SIZE", 16)
    .symbol("NUM_PATCHES", 196, { provenance: "computed", note: "(IMG_HEIGHT // PATCH_SIZE) ** 2" })
    .symbol("PROJECTION_DIM", 64)
    .symbol("NUM_HEADS", 4)
    .symbol("MLP_DIM", 128)
    .symbol("TRANSFORMER_LAYERS", 4)
    .symbol("num_classes", 8, {
      provenance: "from-output",
      note: "len(class_names); the notebook reports 8 classes",
    })
    .model("m_vit", "ViT_scratch", "Mini Vision Transformer trained from scratch", [
      ref(cell, 61, 80, "definition"),
      ref(cell, 83, 83, "call"),
    ])
    .node(
      "n_in",
      "input",
      { shape: [224, 224, 3] },
      { label: "Input 224×224×3", refs: [ref(cell, 63, 63)], trainable: "n/a" },
    )
    .node(
      "n_patches",
      "patchify",
      { patch: "PATCH_SIZE" },
      {
        from: ["n_in"],
        label: "Patches 16×16",
        customOpName: "Patches",
        refs: [ref(cell, 65, 65, "apply"), ref(cell, 1, 20, "definition")],
        trainable: "n/a",
      },
    );

  b.group("g_patch_encoder", "PatchEncoder", {
    kind: "module",
    refs: [ref(cell, 66, 66, "call"), ref(cell, 23, 36, "definition")],
    description: "Projects each patch to 64 numbers and adds a learned position embedding.",
  })
    .node(
      "n_proj",
      "patch_embed",
      { dim: "PROJECTION_DIM" },
      {
        from: ["n_patches"],
        label: "Patch projection → 64",
        refs: [ref(cell, 30, 30), ref(cell, 35, 35, "apply")],
      },
    )
    .node(
      "n_pos",
      "pos_embed",
      { length: "NUM_PATCHES", dim: "PROJECTION_DIM" },
      {
        from: ["n_proj"],
        label: "Position embedding",
        refs: [ref(cell, 31, 31), ref(cell, 34, 35, "apply")],
      },
    );

  b.group("g_blocks", "Transformer block", {
    kind: "repeat",
    repeatSymbol: "TRANSFORMER_LAYERS",
    refs: [ref(cell, 68, 69, "loop"), ref(cell, 39, 58, "definition")],
    description:
      "Pre-norm encoder block: LayerNorm → attention → add, then LayerNorm → MLP (GELU) → add.",
  })
    .node(
      "n_ln1",
      "layernorm",
      { epsilon: 1e-6 },
      { from: ["n_pos"], label: "LayerNorm", refs: [ref(cell, 44, 44)] },
    )
    .node(
      "n_mha",
      "multi_head_attention",
      { numHeads: "NUM_HEADS", keyDim: "PROJECTION_DIM", dropout: 0.1 },
      {
        from: ["n_ln1", "n_ln1"],
        label: "Attention · 4 heads",
        refs: [ref(cell, 45, 47)],
        explanation:
          "Each of the 196 patch tokens attends to every other. key_dim=64 is per head, so the attention space is 256 wide.",
      },
    )
    .node(
      "n_add1",
      "add",
      {},
      {
        from: ["n_mha", { id: "n_pos", kind: "residual" }],
        label: "Add (residual)",
        refs: [ref(cell, 48, 48)],
        trainable: "n/a",
      },
    )
    .node(
      "n_ln2",
      "layernorm",
      { epsilon: 1e-6 },
      { from: ["n_add1"], label: "LayerNorm", refs: [ref(cell, 51, 51)] },
    )
    .node(
      "n_mlp1",
      "dense",
      { units: "MLP_DIM", activation: "gelu" },
      { from: ["n_ln2"], label: "MLP Dense 128 · GELU", refs: [ref(cell, 52, 52)] },
    )
    .node(
      "n_mlp_drop1",
      "dropout",
      { rate: 0.1 },
      { from: ["n_mlp1"], label: "Dropout 0.1", refs: [ref(cell, 53, 53)], trainable: "n/a" },
    )
    .node(
      "n_mlp2",
      "dense",
      { units: "PROJECTION_DIM" },
      { from: ["n_mlp_drop1"], label: "MLP Dense 64", refs: [ref(cell, 54, 54)] },
    )
    .node(
      "n_mlp_drop2",
      "dropout",
      { rate: 0.1 },
      { from: ["n_mlp2"], label: "Dropout 0.1", refs: [ref(cell, 55, 55)], trainable: "n/a" },
    )
    .node(
      "n_add2",
      "add",
      {},
      {
        from: ["n_mlp_drop2", { id: "n_add1", kind: "residual" }],
        label: "Add (residual)",
        refs: [ref(cell, 56, 56)],
        trainable: "n/a",
      },
    );

  b.group("g_head", "Classifier", { kind: "head" })
    .node(
      "n_ln_out",
      "layernorm",
      { epsilon: 1e-6 },
      { from: ["n_add2"], label: "LayerNorm", refs: [ref(cell, 71, 71)] },
    )
    .node(
      "n_gap",
      "global_avg_pool",
      {},
      {
        from: ["n_ln_out"],
        label: "Average over tokens",
        refs: [ref(cell, 72, 72)],
        trainable: "n/a",
      },
    )
    .node(
      "n_drop1",
      "dropout",
      { rate: 0.3 },
      { from: ["n_gap"], label: "Dropout 0.3", refs: [ref(cell, 73, 73)], trainable: "n/a" },
    )
    .node(
      "n_fc1",
      "dense",
      { units: 128, activation: "gelu" },
      { from: ["n_drop1"], label: "Dense 128 · GELU", refs: [ref(cell, 75, 75)] },
    )
    .node(
      "n_drop2",
      "dropout",
      { rate: 0.3 },
      { from: ["n_fc1"], label: "Dropout 0.3", refs: [ref(cell, 76, 76)], trainable: "n/a" },
    )
    .node(
      "n_out",
      "dense",
      { units: "num_classes", activation: "softmax" },
      { from: ["n_drop2"], label: "Dense 8 · softmax", refs: [ref(cell, 77, 77)] },
    )
    .outputs("n_out");
  return b.build();
}

export const EXAMPLE_IRS: Record<string, () => WireIR> = {
  alexnet: alexnetIR,
  vgg16: vgg16IR,
  "mini-vit": miniVitIR,
};
