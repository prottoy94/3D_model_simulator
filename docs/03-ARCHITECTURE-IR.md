# 03 — Architecture IR, Op Catalog, Shape Engine

The **Architecture IR** is the contract between notebook understanding (heuristics + Fable 5.2)
and everything that renders or simulates. It has two forms:

| Form | Produced by | Purpose | Location |
|---|---|---|---|
| **Wire IR** | Fable 5.2 (structured output), Forge prompt, fast-path parser | A transport format that fits the structured-output constraints | `src/core/ir/wire.ts` |
| **Runtime IR** | `normalize()` + zoo expansion + shape engine | Typed, verified, indexed, ready for layout and simulation | `src/core/ir/runtime.ts` |

Design constraints that come from Claude structured outputs (see doc 05):
- **No recursive schemas.** Hierarchy is expressed with a flat `groups[]` array and `parentId`.
- **Every object has `additionalProperties: false`.** So free-form maps are not allowed, and
  layer parameters are an **array of `{key, value}` pairs**.
- No `minimum`/`maximum`/`minLength`. The SDK strips them, and Zod re-checks them client-side.

---

## 1. Wire IR (v1.0)

Specification in TypeScript notation; implement it as Zod in `src/core/ir/wire.ts`.

```ts
type CodeRef = {
  cell: number;              // notebook cell index (0-based, as in the file)
  startLine: number;         // 1-based, inclusive, within the cell
  endLine: number;           // 1-based, inclusive
  role: "declare"            // where the layer object is constructed (Keras call, nn.Conv2d in __init__)
      | "apply"              // where it is applied to a tensor (x = self.conv(x), layer(x))
      | "definition"         // def/class body that defines a reusable block
      | "call"               // call site that instantiates a block (transformer_block(...))
      | "loop"               // loop that repeats a block (for _ in range(N))
      | "config";            // constant / hyper-parameter definition
};

type ParamValue = string | number | boolean | number[];

type WireIR = {
  irVersion: "1.0";
  scope: "notebook" | "snippet" | "forge";
  title: string;                         // e.g. "Micro-organism classifiers: AlexNet & VGG16"
  summary: string;                       // 1–3 plain-English sentences
  framework: "keras" | "pytorch" | "tensorflow" | "jax" | "sklearn"
           | "huggingface" | "ultralytics" | "mixed" | "other";
  domain: "cv" | "nlp" | "tabular" | "audio" | "timeseries" | "multimodal" | "other";
  task: "classification" | "detection" | "segmentation" | "regression" | "generation"
      | "sequence-labeling" | "clustering" | "embedding" | "multi-task" | "other";

  symbols: Array<{
    name: string;                        // IMG_HEIGHT, PATCH_SIZE, num_classes
    value: string;                       // JSON-encoded literal: "224", "[224,224,3]", "\"relu\""
    provenance: "literal" | "computed" | "from-output" | "assumed";
    ref: CodeRef | null;                 // where it is defined (or the output cell it was read from)
    note: string;                        // e.g. "len(class_names); 8 classes per output of cell 6"
  }>;

  models: Array<{
    id: string;                          // "m_alexnet"
    name: string;                        // "AlexNet"
    description: string;
    rootGroupId: string;
    inputNodeIds: string[];
    outputNodeIds: string[];
    isPrimary: boolean;                  // the model a user most likely wants first
    variantOf: string | null;            // e.g. inference vs train_forward paths
  }>;

  groups: Array<{
    id: string;
    parentId: string | null;             // null only for a model root group
    modelId: string;
    label: string;                       // "Block 3", "Transformer block", "RPN head"
    kind: "model" | "stage" | "block" | "repeat" | "module" | "backbone" | "head" | "branch";
    repeatCount: number | null;          // kind=repeat: 4 for TRANSFORMER_LAYERS
    repeatSymbol: string | null;         // "TRANSFORMER_LAYERS"
    zooKey: string | null;               // kind=backbone: "keras.applications.EfficientNetB0"
    zooArgs: Array<{ key: string; value: ParamValue }>;   // include_top=false, input_shape, ...
    refs: CodeRef[];
    description: string;
  }>;

  nodes: Array<{
    id: string;                          // stable, readable: "n_conv1"
    groupId: string;
    op: OpKind;                          // enum from the op catalog (§3)
    customOpName: string | null;         // op="custom": the user's class name (e.g. "ROIPoolingLayer")
    label: string;                       // "Conv 1 · 96@11×11 /4"
    params: Array<{ key: string; value: ParamValue }>;   // canonical keys per op (catalog)
    trainable: "yes" | "no" | "partial" | "n/a";
    claimedOutputShape: Array<number | string> | null;  // the LLM's belief, native layout; "B" = batch
    refs: CodeRef[];
    explanation: string;                 // one sentence, specific to this node
    confidence: "high" | "medium" | "low";
  }>;

  edges: Array<{
    id: string;
    from: string;                        // node id
    to: string;                          // node id
    toPort: number;                      // input slot, ordered (concat order; MHA query=0, value=1, key=2)
    kind: "data" | "residual" | "skip" | "broadcast" | "control";
  }>;

  pipelines: Array<{                     // non-NN stages around/among models (detection, classical ML)
    id: string;
    label: string;                       // "R-CNN inference"
    stages: Array<{
      id: string;
      label: string;                     // "Selective Search proposals"
      kind: "preprocess" | "algorithm" | "model" | "postprocess" | "loss";
      algorithm: AlgorithmKind | null;   // "selective_search" | "nms" | "anchor_generation" | ...
      modelId: string | null;            // kind=model
      params: Array<{ key: string; value: ParamValue }>;
      refs: CodeRef[];
      description: string;
    }>;
  }>;

  training: {
    optimizer: string | null;            // "Adam"
    learningRate: number | null;
    loss: string | null;                 // "categorical_crossentropy"
    metrics: string[];
    epochs: number | null;
    batchSize: number | null;
    phases: Array<{                      // transfer learning: frozen → fine-tune
      name: string;                      // "Phase 1: head only"
      frozenGroupIds: string[];
      unfreezeRule: string | null;       // "last 20 layers of base_model", "last 2 encoder blocks"
      learningRate: number | null;
      ref: CodeRef | null;
    }>;
  } | null;

  warnings: string[];                    // things the model could not resolve
};
```

### Conventions

- **ids:** `m_*` models, `g_*` groups, `n_*` nodes, `e_*` edges, `p_*` pipelines. They are unique
  across the whole IR.
- **Repeats:** a `repeat` group contains **one canonical instance** of the block's nodes.
  Normalization **unrolls** it into N instances (ids suffixed `#1..#N`) and keeps a link back to
  the canonical template, so Lens can map the function definition to every instance.
- **Pretrained backbones:** emit **one** `backbone` group with a `zooKey` and `zooArgs`, plus a
  single placeholder node `op: "backbone"`. Do not try to list 230 EfficientNet layers. The zoo
  expands it (§5).
- **Fused activations:** `Conv2D(..., activation="relu")` becomes one `conv2d` node with
  `activation: "relu"`. The simulator shows the activation as a sub-phase. Separate `ReLU()`
  layers become their own `activation` nodes.
- **Functional ops in `forward`** (`torch.cat`, `F.interpolate`, `x.mean(1)`, `x * torch.sigmoid(a)`)
  become nodes too (`concat`, `upsample`, `reduce_mean`, `multiply`, `sigmoid`), each with an `apply` ref.
- **Multiple forward paths** (`forward` vs `train_forward`, `tasks=("cls","det")`) are modelled as
  the inference path. The alternatives go in `warnings` or `variantOf` models.

---

## 2. Runtime IR

Produced by `normalize(wire) → expand(zoo) → verify(shapeEngine)`. It adds the following:

```ts
type Dim = number | { sym: string };                  // { sym: "B" } batch, { sym: "T" } sequence, { sym: "R" } RoIs
type Layout = "NHWC" | "NCHW" | "NLC" | "NCL" | "NC" | "NL" | "NTC" | "boxes" | "scalar" | "any";
type TensorShape = { dims: Dim[]; layout: Layout };

type RuntimeNode = Omit<WireNode, "params"> & {
  params: OpParams[op];                              // typed via catalog Zod schema, defaults applied
  inShapes: TensorShape[];
  outShape: TensorShape | null;                      // null if inference failed
  paramCount: { trainable: number; nonTrainable: number };   // framework-aware (see §4.3)
  flops: number;                                     // multiply-adds × 2, per sample
  status: "ok" | "mismatch" | "unverified" | "error";
  issues: Array<{ code: string; message: string; severity: "info" | "warn" | "error" }>;
  provenance: "ai" | "zoo" | "ai-synth" | "heuristic" | "user";
  templateId: string | null;                         // canonical node id if this is an unrolled repeat instance
  summaryRow: SummaryRow | null;                     // matched row from model.summary(), if any
};

type RuntimeIR = Omit<WireIR, "nodes" | "groups"> & {
  nodes: Record<string, RuntimeNode>;
  groups: Record<string, RuntimeGroup>;              // + stats: { params, flops, depth }, + expandedByDefault
  index: {
    topoOrder: Record<string /*modelId*/, string[]>;
    children: Record<string /*groupId*/, { groups: string[]; nodes: string[] }>;
    inEdges: Record<string, string[]>;
    outEdges: Record<string, string[]>;
    codeIntervals: IntervalTree<{ nodeId?: string; groupId?: string; role: CodeRef["role"] }>;  // for Lens
  };
  verification: VerificationReport;                  // §6
  assumptions: Array<{ symbol: string; value: string; source: string; editable: true }>;
};
```

Normalization steps, in order:
1. Zod-parse the wire IR and reject structural garbage.
2. Substitute symbols, so a `params` value of `"PATCH_SIZE"` becomes `16`, with provenance kept.
3. Parse params with the op's catalog schema: convert `{key, value}[]` to a typed object and
   apply framework defaults (Keras `padding="valid"`, PyTorch `padding=0`).
4. Unroll repeat groups.
5. Check graph integrity: no dangling edges, every non-input node has inputs, no cycles in the
   data graph (recurrence is internal to `lstm`/`gru`/`rnn` nodes), and every node's group exists.
6. Expand zoo backbones.
7. Run the shape engine (§4) and verification (§6).
8. Build the indexes.

---

## 3. The Op Catalog

The single source of truth for what an operation *is*. It lives in `src/core/ops/<family>.ts`
and is aggregated in `src/core/ops/index.ts`. The `OpKind` enum in the wire schema is generated from it.

```ts
interface OpSpec<P> {
  op: OpKind;                                   // "conv2d"
  family: OpFamily;                             // drives color + default glyph
  title: string;                                // "2D Convolution"
  aliases: {                                    // used by heuristics, the fast-path parser and prompts
    keras?: string[];                           // ["Conv2D", "layers.Conv2D", "keras.layers.Conv2D"]
    pytorch?: string[];                         // ["nn.Conv2d"]
    functional?: string[];                      // ["F.conv2d", "tf.nn.conv2d"]
  };
  params: z.ZodType<P>;                         // canonical param schema
  defaults: { keras: Partial<P>; pytorch: Partial<P> };
  arity: { min: number; max: number | null };   // number of inputs
  inferShape(inputs: TensorShape[], p: P, ctx: ShapeCtx): TensorShape | ShapeError;
  countParams(inputs: TensorShape[], p: P, ctx: ShapeCtx): { trainable: number; nonTrainable: number };
  flops(inputs: TensorShape[], out: TensorShape, p: P): number;
  glyph: GlyphKind;                             // "volume" | "neuron-column" | "matrix" | "ring" | "funnel" | "gate" | ...
  visualizer: VisualizerId;                     // doc 04
  explain(p: P, io: { in: TensorShape[]; out: TensorShape }): string;
  codegen: { pytorch(p: P, names: CodegenNames): string; keras(p: P, names: CodegenNames): string };
}
```

### v1 op list (grouped by family)

| Family | Ops | Color token |
|---|---|---|
| io | `input`, `output` | `--op-io` |
| conv | `conv1d`, `conv2d`, `conv3d`, `depthwise_conv2d`, `separable_conv2d`, `conv_transpose2d` | `--op-conv` (cyan) |
| pool | `maxpool1d`, `maxpool2d`, `avgpool2d`, `global_avg_pool`, `global_max_pool`, `adaptive_avg_pool2d` | `--op-pool` (violet) |
| norm | `batchnorm`, `layernorm`, `groupnorm`, `instancenorm` | `--op-norm` (teal) |
| activation | `relu`, `leaky_relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `softmax`, `hardswish` | `--op-act` (lime) |
| linear | `dense`, `embedding`, `patchify`, `patch_embed`, `pos_embed`, `cls_token` | `--op-linear` (amber) |
| attention | `multi_head_attention`, `channel_attention`, `spatial_attention`, `squeeze_excite` | `--op-attn` (magenta) |
| recurrent | `rnn`, `lstm`, `gru`, `bidirectional` | `--op-recur` (sky) |
| reshape | `flatten`, `reshape`, `permute`, `slice`, `squeeze`, `unsqueeze`, `upsample`, `time_distributed` (wrapper) | `--op-shape` (slate) |
| merge | `add`, `concat`, `multiply`, `matmul`, `reduce_mean`, `reduce_max` | `--op-merge` (white) |
| regularization | `dropout`, `drop_path` | `--op-reg` (grey, ghosted) |
| detection | `anchor_generator`, `roi_pool`, `roi_align`, `box_decode`, `nms` | `--op-det` (orange) |
| composite | `backbone` (zoo placeholder) and macro-expanded blocks (`mbconv`, `inverted_residual`, `bottleneck`, `convnext_block`, `transformer_encoder`) | per children |
| classical | `estimator` (sklearn: algorithm in params) | `--op-classic` (gold) |
| custom | `custom` (an unknown user layer: label, description, declared output shape) | `--op-custom` (rose outline) |

`AlgorithmKind` (pipeline stages): `selective_search`, `iou`, `nms`, `anchor_generation`,
`anchor_labeling`, `box_encode`, `box_decode`, `crop_resize`, `tokenize`, `train_test_split`,
`gradient_descent`, `tree_split`, `kmeans_step`.

### Canonical param keys (examples)

| Op | Keys |
|---|---|
| `conv2d` | `filters`, `kernel` `[kh,kw]`, `stride` `[sh,sw]`, `padding` (`"valid"`/`"same"`/`[ph,pw]`), `dilation`, `groups`, `bias`, `activation` |
| `maxpool2d` | `pool` `[ph,pw]`, `stride` (default = pool in Keras), `padding` |
| `dense` | `units`, `bias`, `activation` (applies to the **last axis**; works on N-D input) |
| `multi_head_attention` | `numHeads`, `keyDim` (**per head**, Keras semantics), `valueDim`, `outputDim`, `dropout`, `selfAttention` |
| `patchify` | `patch` `[ph,pw]` |
| `pos_embed` | `length`, `dim`, `learned` |
| `dropout` | `rate` (visual: training vs inference toggle) |
| `estimator` | `algorithm` (`"random_forest"`...), plus the algorithm's hyper-parameters |

---

## 4. Shape engine (`src/core/shapes`)

### 4.1 Propagation
Walk the nodes in topological order per model. `inShapes` come from incoming edges ordered by
`toPort`. Call `catalog[op].inferShape`. On a `ShapeError`, mark the node `error` and continue
with the LLM's `claimedOutputShape` (if any) so downstream nodes can still be shown. Mark
everything downstream `unverified`.

### 4.2 Layout handling
Shapes are stored in the framework's **native layout** (Keras NHWC, PyTorch NCHW), so the UI
shows what the user's code means. Catalog shape functions never index raw dims. They use helpers
(`spatial(shape)`, `channels(shape)`, `withSpatial(shape, [h, w])`, `seqLen(shape)`,
`featureDim(shape)`) that read the `layout` tag. Batch is always `{sym:"B"}`.

### 4.3 Formulas (must have unit tests)

| Op | Rule |
|---|---|
| conv (PyTorch) | `out = floor((in + 2p − d·(k−1) − 1) / s) + 1` |
| conv / pool (Keras `valid`) | `out = floor((in − k) / s) + 1` |
| conv / pool (Keras `same`) | `out = ceil(in / s)` |
| Keras pooling default | `strides = pool_size` when not given |
| `dense` params | `in_last × units + (bias ? units : 0)` |
| `conv2d` params | `kh·kw·(cin/groups)·filters + (bias ? filters : 0)` |
| `batchnorm` params | **Keras:** `4C` = 2C trainable (γ, β) + 2C non-trainable (moving mean/var). **PyTorch:** `2C` parameters; running stats are buffers and are not counted as params |
| `layernorm` params | `2 × normalized_dim` |
| `multi_head_attention` params (Keras) | `3 × (D·h·k + h·k) + (h·k·D_out + D_out)`, where `k = keyDim` per head, `h = numHeads`, `D` = query feature dim |
| `embedding` params | `vocab × dim` |
| `patchify` | `(H/p)·(W/p)` tokens of size `p·p·C` |
| `global_avg_pool` | `(B,H,W,C) → (B,C)` |
| FLOPs (conv) | `2 · H_out·W_out · kh·kw·(cin/groups) · cout` |
| FLOPs (dense) | `2 · in · units` (× tokens for N-D input) |

### 4.4 Ground-truth checks (from the reference notebooks)
These come from the notebooks' saved `model.summary()` output (doc 07) and are permanent unit tests:
- AlexNet conv1: `(B, 54, 54, 96)`, 34,944 params; first pool → `(B, 26, 26, 96)`; total **46,781,192**
  (trainable 46,780,488 + non-trainable 704 from BatchNorm moving stats).
- VGG16 variant (GAP head, 8 classes): total **14,981,448**.
- Mini-ViT: PatchEncoder 61,760; MHA (4 heads, keyDim 64, D 64) **66,368**; LayerNorm 128; total **404,040**.

---

## 5. Model zoo (`src/core/zoo`)

- Templates are **IR fragments** (groups + nodes + edges with canonical params), stored as
  versioned JSON: `src/core/zoo/templates/<key>.json`.
- **Keys:** `keras.applications.EfficientNetB0`, `keras.applications.MobileNetV2`,
  `keras.applications.ResNet50`, `keras.applications.VGG16`, `keras_hub.ViTBackbone:vit_base_patch16_224_imagenet`,
  `torchvision.models.convnext_tiny`, `torchvision.models.resnet50`, `ultralytics.yolov8n`, ...
- Each template declares its parameters (input shape, `include_top`, width multiplier) and a
  pure `instantiate(args) → IRFragment` (e.g., for `include_top=False`, drop the classifier).
- **Hierarchy matters:** templates are grouped into stages and blocks (EfficientNetB0: stem →
  7 MBConv stages → top conv) so they collapse nicely.
- **Producing templates:** an optional developer script, `tools/zoo-export/` (Python, run locally
  once, outputs committed), loads the real `keras.applications` / `torchvision` model and exports
  its graph (`model.get_config()` / `torch.fx`) into IR JSON. The script also checks total params
  against the library. **This is the only place Python runs, and it never runs user code.**
- **Unknown backbones:** `/api/ai/zoo` asks Fable 5.2 to synthesize a template from its knowledge.
  The result is flagged `provenance: "ai-synth"` and shown with a dashed outline and a badge until
  the user accepts it.
- Frozen state: training `phases` mark zoo groups as frozen per phase (EfficientNet phase 2 = the
  last `FINE_TUNE_LAYERS` layers unfrozen).

---

## 6. Verification (`VerificationReport`)

Sources of truth, best first:
1. **Parsed notebook outputs:** Keras 2 summary (plain text), **Keras 3 summary** (rich box-drawing
   table with ANSI codes, wrapped and `…`-truncated names, split across several `display_data`
   outputs), torchinfo tables, `print(model)` module trees (structure only), and printed totals.
2. **Shape engine** computations.
3. **LLM claims** (`claimedOutputShape`).

Matching: align summary rows with IR nodes by order within a model, plus layer type
(`Conv2D` ↔ `conv2d`), plus name where available. Composite Keras layers (e.g., a custom
`PatchEncoder`) match a **group** or node sequence, using the group's aggregated params.

```ts
type VerificationReport = {
  perModel: Record<string, {
    totals: { computed: ParamTotals; summary: ParamTotals | null; match: boolean | null };
    rows: Array<{ nodeOrGroupId: string; summaryShape: string | null; computedShape: string;
                  summaryParams: number | null; computedParams: number; status: "match" | "mismatch" | "no-summary" }>;
  }>;
  llmDisagreements: Array<{ nodeId: string; claimed: string; computed: string }>;
  repairAttempts: number;
};
```

Known benign differences: Keras sometimes reports dynamic dims (`(None, None, 768)` for the custom
`Patches` layer). Where the summary has `None` and the engine has a number, prefer the engine's
value and treat it as a match.

### Repair loop
If any `mismatch` or `error` remains, send the **diff list** (not the whole IR) back to Fable 5.2
as an appended turn in the same conversation. It answers with a `RepairPatch`:

```ts
type RepairPatch = {
  ops: Array<{
    action: "update_node" | "add_node" | "remove_node" | "add_edge" | "remove_edge"
          | "update_group" | "set_symbol";
    targetId: string;
    node: WireNode | null; edge: WireEdge | null; group: WireGroup | null;
    symbol: { name: string; value: string } | null;
    reason: string;
  }>;
};
```

Apply the patch and re-verify, at most **2** times. Whatever remains is shown with ⚠ badges and
editable assumptions.

---

## 7. Worked examples (abridged wire IR)

### AlexNet (Keras Sequential), first layers

```json
{
  "symbols": [
    { "name": "IMG_HEIGHT", "value": "224", "provenance": "literal",
      "ref": { "cell": 4, "startLine": 10, "endLine": 10, "role": "config" }, "note": "" },
    { "name": "num_classes", "value": "8", "provenance": "from-output",
      "ref": { "cell": 12, "startLine": 22, "endLine": 22, "role": "config" },
      "note": "len(class_names); outputs of cells 6 and 12 report 8 classes" }
  ],
  "models": [{ "id": "m_alexnet", "name": "AlexNet", "rootGroupId": "g_alexnet",
               "inputNodeIds": ["n_in"], "outputNodeIds": ["n_fc3"], "isPrimary": true,
               "description": "AlexNet adapted with BatchNorm, trained from scratch", "variantOf": null }],
  "nodes": [
    { "id": "n_in", "groupId": "g_alexnet", "op": "input", "label": "Input 224×224×3",
      "params": [{ "key": "shape", "value": [224, 224, 3] }], "trainable": "n/a",
      "claimedOutputShape": ["B", 224, 224, 3],
      "refs": [{ "cell": 14, "startLine": 12, "endLine": 12, "role": "declare" }],
      "explanation": "RGB image resized to 224×224.", "confidence": "high", "customOpName": null },
    { "id": "n_conv1", "groupId": "g_alexnet", "op": "conv2d", "label": "Conv 1 · 96@11×11 /4",
      "params": [{ "key": "filters", "value": 96 }, { "key": "kernel", "value": [11, 11] },
                 { "key": "stride", "value": [4, 4] }, { "key": "padding", "value": "valid" },
                 { "key": "activation", "value": "relu" }],
      "trainable": "yes", "claimedOutputShape": ["B", 54, 54, 96],
      "refs": [{ "cell": 14, "startLine": 12, "endLine": 12, "role": "declare" }],
      "explanation": "Large 11×11 filters with stride 4 shrink the image fast while detecting coarse patterns.",
      "confidence": "high", "customOpName": null }
  ],
  "edges": [{ "id": "e1", "from": "n_in", "to": "n_conv1", "toPort": 0, "kind": "data" }]
}
```

### Mini-ViT transformer block (repeat group)

```json
{
  "groups": [
    { "id": "g_blocks", "parentId": "g_vit", "modelId": "m_vit", "label": "Transformer blocks",
      "kind": "repeat", "repeatCount": 4, "repeatSymbol": "TRANSFORMER_LAYERS", "zooKey": null, "zooArgs": [],
      "refs": [{ "cell": 16, "startLine": 68, "endLine": 69, "role": "loop" },
               { "cell": 16, "startLine": 39, "endLine": 58, "role": "definition" }],
      "description": "Pre-norm encoder block: LN → MHA → +residual → LN → MLP(GELU) → +residual" }
  ],
  "nodes": [
    { "id": "n_ln1", "groupId": "g_blocks", "op": "layernorm", "label": "LayerNorm",
      "params": [{ "key": "epsilon", "value": 1e-6 }], "...": "..." },
    { "id": "n_mha", "groupId": "g_blocks", "op": "multi_head_attention", "label": "MHA · 4 heads",
      "params": [{ "key": "numHeads", "value": 4 }, { "key": "keyDim", "value": 64 },
                 { "key": "dropout", "value": 0.1 }, { "key": "selfAttention", "value": true }],
      "explanation": "Each of the 196 patch tokens attends to every other; key_dim=64 is per head, so the attention space is 256 wide.",
      "...": "..." }
  ]
}
```

---

## 8. Versioning

- `irVersion` follows semver. A minor bump adds optional fields, and a major bump requires a migration in
  `src/core/ir/migrations/`. Cached IR in IndexedDB carries its version and migrates on load.
- Prompt versions (doc 05) and IR versions are tracked separately. Both are part of the AI cache key.
