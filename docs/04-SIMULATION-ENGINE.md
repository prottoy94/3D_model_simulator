# 04 — Simulation Engine, Micro-Visualizers, Lens

The simulation turns a static architecture into a **process you can watch**. A sample input
travels through the model one operation at a time. Every step shows *what the operation does*
(a micro-animation in the inspector) and *where it happens* (pulses and lighting in the 3D
viewport), all tied to the code that created it.

## 1. Concepts

| Term | Meaning |
|---|---|
| **SimPlan** | A compiled, ordered list of steps for one model (or a Lens subgraph). Pure data. |
| **Step** | One node (or one collapsed group, as a *macro-step*) executing: inputs → output. |
| **Clock** | Playback time `t` in ms, plus `playing`, `speed` and `loop`. A single source of time for everything. |
| **Phase** | A sub-interval of a step's duration that a visualizer animates (e.g., conv: *sweep* → *assemble* → *activate*). |
| **Tensor backend** | Computes the real (or symbolic) values that the visualizers display. |
| **TensorView** | A downsampled, display-ready view of a tensor (channels sampled, normalized, stats). |

## 2. SimPlan compiler (`src/core/sim/compile.ts`)

```ts
type SimStep = {
  id: string;
  nodeId: string | null;          // null for a macro-step
  groupId: string | null;         // set for a macro-step (collapsed group) or as context
  inputs: string[];               // producer step ids
  startMs: number;
  durationMs: number;             // from catalog weight × global pacing (conv longer than relu)
  visualizer: VisualizerId;
  phases: Array<{ name: string; from: number; to: number }>;  // fractions 0..1 of the step
  narration: string;              // catalog explain(), enriched by the IR node's explanation
};
type SimPlan = { modelId: string; steps: SimStep[]; totalMs: number; focus: FocusSpec | null };
```

- The order is topological. Parallel branches (Inception, multi-head, FPN levels, detection
  heads) are scheduled **concurrently** in the same time window, so the viewport shows parallelism.
- A **collapsed group is a macro-step**: it plays as a single "pass-through" glow with a summary
  narration ("MBConv stage 4: 3 blocks, 112×… → …"). Expanding the group while paused recompiles
  the plan, and the clock maps to the equivalent position.
- **Unrolled repeats** play every instance. The pacing compresses instances 2..N (fast-forward
  shimmer) unless the user expands the repeat group.
- `stateAt(plan, t)` is a **pure function** that returns active steps and phase progress. It makes
  scrubbing, stepping and deterministic screenshots trivial.

## 3. Clock and rendering loop

- The clock lives in `simStore`, advanced by one `requestAnimationFrame` loop
  (`t += dt × speed` while `playing`).
- R3F components read the clock with `useSimStore.getState()` inside `useFrame` and mutate
  refs (materials, instance colors, particle uniforms). **No React re-render per frame.**
- Inspector visualizers are 2D Canvas/WebGL components. They subscribe to the *active step id*
  (rare re-renders) and read `progress` per frame through refs.
- **Controls:** play/pause (Space), step forward/back (→/←, which jumps to step boundaries),
  scrub (timeline drag), speed (0.25× to 4×), loop, and "jump to node" (click a node).

## 4. Compute tiers (tensor backends)

```ts
interface TensorBackend {
  readonly tier: "symbolic" | "random-init" | "real-weights";
  prepare(ir: RuntimeIR, modelId: string, input: InputSample): Promise<void>;
  run(step: SimStep): Promise<StepTensors>;      // inputs/outputs as backend handles
  view(handle: TensorHandle, opts: ViewOpts): TensorView;   // downsample for display
  dispose(): void;
}
```

| Tier | What | When | Label shown in UI |
|---|---|---|---|
| **Tier 0: symbolic** | No numeric compute. Activations are procedural textures derived from the input image (e.g., blurred/edge-filtered variants) at the correct shapes | Huge models, unsupported ops, no WebGL/WebGPU | "Symbolic activations: shapes are exact, values are illustrative" |
| **Tier 1: random-init** (default) | A real forward pass with tfjs: an **IR interpreter** walks the graph and calls tfjs ops per node. Weights use He/Glorot init with a fixed seed. Optional **teaching kernels** for the first conv layer (Sobel, Gabor, color blobs) make early feature maps readable | Default for models within the compute budget | "Real computation · random-init weights (not your trained model)" |
| **Tier 2: real-weights** (P2) | The same interpreter, with weights loaded from an uploaded ONNX/`.keras` file mapped onto IR nodes | The user provides weights | "Your trained weights" |

- **IR interpreter:** `src/core/sim/interpreter/` has one executor per op family, implemented with
  tfjs primitives (`conv2d`, `depthwiseConv2d`, `matMul`, `maxPool`, `avgPool`, `batchNorm`,
  `softmax`, `einsum`/`matMul` for attention, `image.cropAndResize` for RoI pooling). Ops without
  an executor fall back to symbolic for that node only.
- **Budget:** estimate FLOPs and parameter memory from the runtime IR. Above about 20 GFLOPs or
  500 MB of parameters (tunable), default to Tier 0 and offer "Run real compute anyway".
- **Inputs:** bundled CC0 sample images (`public/samples/`), user upload (image, CSV row, text),
  and a synthetic pattern generator (checkerboard, gradients, shapes, useful for showing what conv
  kernels respond to). NLP: a simple whitespace/char tokenizer for demos. A real tokenizer comes
  later via the zoo.
- **TensorView rules:** pick ≤ 16 channels by activation energy, downsample to ≤ 128×128,
  normalize per channel, and record min/max/mean/std and a sparsity metric (share of zeros after ReLU).
  Dispose of tfjs tensors as soon as their views exist.

## 5. Viewport choreography (3D)

For each active step:
1. **Approach:** pulses (GPU particles) travel along the incoming edges. Their color is the producer's family color.
2. **Ignite:** the node's glass volume brightens, a scanline sweeps through it, and its label expands into a HUD tag (`54×54×96 · 34,944 params`).
3. **Transform:** the volume's dimensions morph from the input shape to the output shape.
   Feature-map thumbnails from TensorViews appear as "slices" stacked inside the volume (a
   sampled set of channels, like film frames).
4. **Emit:** pulses leave along the outgoing edges.

Special motions: **Flatten** unrolls the 3D volume into a long ribbon. **Concat** shows streams
docking side by side. **Add (residual)** shows two streams merging with a flash. **Dropout** blacks
out neurons (training-mode toggle). **Frozen** nodes have frost and a lock glyph, and pulses pass
through them without a "learning shimmer". **Repeat ×N** shows ghost copies that light up in sequence.

Camera modes: *Orbit* (free), *Follow* (a cinematic dolly that tracks the active step), *Overview*
(fit the model), *Focus* (Lens subgraph). Transitions are spring-animated.

## 6. Micro-visualizer catalog (inspector)

Contract:

```ts
type VisualizerProps = {
  node: RuntimeNode;
  inputs: TensorView[];
  output: TensorView | null;
  progressRef: React.MutableRefObject<number>;   // 0..1 within the step, updated per frame
  phase: string;
  quality: "low" | "high";
  illustrative: boolean;                          // true for Tier 0, so the label is shown
};
type VisualizerSpec = { id: VisualizerId; ops: OpKind[]; phases: string[]; component: React.FC<VisualizerProps> };
```

| Visualizer | Ops | What the user sees |
|---|---|---|
| `conv-sweep` | conv1d/2d/3d, depthwise, separable, conv_transpose | The input feature map with a glowing k×k kernel window sliding by the stride. A live dot product (window ⊙ kernel → sum + bias). The output pixel fills in. Then a fast sweep across all output channels. Padding cells are shown as a dotted border. Depthwise shows per-channel kernels; separable shows the depthwise then pointwise passes. |
| `pool-window` | maxpool, avgpool | The window slides. Max: the winning cell flashes and is "picked". Avg: cells average together. |
| `global-pool` | global_avg_pool, global_max_pool, adaptive_avg_pool | Each channel's map collapses into one glowing bead, forming a vector. |
| `activation-curve` | relu, gelu, silu, sigmoid, tanh, leaky_relu, hardswish | The function curve, with sample values traveling along it. A heatmap before and after. ReLU clips negatives to black, with a sparsity readout. |
| `softmax-bars` | softmax (and final classifier) | Logits become exponentials, then normalized probability bars, with the predicted class highlighted (class names from symbols, e.g., the 8 micro-organism classes). |
| `norm-histogram` | batchnorm, layernorm, groupnorm | Value histogram before → standardized (μ=0, σ=1) → scaled by γ, shifted by β. BN shows batch statistics (with train vs inference running stats); LN shows per-token statistics. |
| `dense-matrix` | dense | Input vector × weight matrix (a sampled grid) → output vector. Neuron columns light up. For N-D input, shows the "same Dense applied to every token". |
| `flatten-unroll` | flatten, reshape, permute | The H×W×C volume unrolls into a vector (or the axes rotate for permute), with index mapping arrows. |
| `dropout-mask` | dropout, drop_path | A random mask in training mode; an identity pass in inference mode (toggle). |
| `merge-streams` | add, concat, multiply | Two tensors align. Add sums element-wise with a flash; concat stacks channels; multiply gates (an attention mask × features). |
| `patchify-grid` | patchify, patch_embed | The image is sliced into a P×P grid; tiles fly into a sequence and are flattened (16×16×3 = 768) and projected. |
| `pos-embed` | pos_embed, cls_token | A positional vector field is added to the tokens (sinusoid or learned heatmap); the CLS token is prepended. |
| `attention-heads` | multi_head_attention | Tokens → Q/K/V projections → the QKᵀ score matrix (downsampled heatmap) → scaling → softmax → weighted sum of V. Head tabs. Click a query token to see its attention row as a heatmap overlaid on the image patches (ViT). |
| `channel-attention` | channel_attention, squeeze_excite | Avg/max-pool → MLP → sigmoid weights per channel → channels brighten or dim. |
| `spatial-attention` | spatial_attention | Mean/max across channels → conv → sigmoid mask over H×W → applied to the features. |
| `recurrent-unroll` | rnn, lstm, gru, bidirectional | The cell unrolled over time steps. LSTM gates (forget/input/output) animate, and the cell state flows along a "conveyor belt". |
| `embedding-lookup` | embedding | Token ids → rows looked up in the embedding table → vectors. |
| `upsample` | upsample | Nearest/bilinear interpolation grid. |
| `anchors` | anchor_generator | 9 anchors (3 scales × 3 ratios) at one location, then tiled across the feature grid (28×28×9 = 7,056). |
| `roi-pool` | roi_pool, roi_align | A proposal box projected onto the feature map → divided into a 7×7 grid → max/bilinear sample per bin. |
| `box-decode` | box_decode | Anchor + (tx, ty, tw, th) deltas → the refined box, animated. |
| `nms` | nms | Boxes sorted by score; the best is kept; boxes with IoU > threshold fade out; repeat. An IoU readout on each comparison. |
| `pipeline-algorithm` | pipeline stages (selective search, IoU, crop-resize, tokenize, ...) | Algorithm-specific mini-scenes on the input image or data. |
| `estimator-*` | estimator (classical ML; later) | Decision boundary formation, tree traversal, k-means centroid updates, a gradient-descent ball on a loss surface. |
| `custom-box` | custom | A black box with the declared in/out shapes and the LLM's description. An honest "we can't simulate the inside of this custom layer yet". |

Every visualizer shows: the op's formula (KaTeX, small), the concrete numbers for this node
(e.g., `floor((224 − 11) / 4) + 1 = 54`), and the code ref chip (click → editor).

## 7. Lens: select code → simulate that part (`src/core/lens`)

### 7.1 Mapping a selection to IR
1. The editor emits selections as `(cell, startLine, endLine)` ranges (several are allowed).
2. Query `index.codeIntervals` for every ref that overlaps the selection.
3. Resolve the hits with a **priority** order:
   - `declare` / `apply` refs → those **nodes**.
   - `definition` refs (a `def transformer_block` body, a `class ChannelAttention` body) → the
     **template**: all instances light up, and the first instance becomes the canonical focus with a "×N" badge.
   - `loop` refs → the whole **repeat group**.
   - `call` refs → that specific instance.
   - `config` refs (selecting `PATCH_SIZE = 16`) → every node whose params use that symbol
     (patchify, pos_embed, ...), with the dependency highlighted.
4. Expand to a **closed subgraph**: the selected nodes plus, optionally, the nodes between them
   (if the user selected two lines that bracket a chain).

### 7.2 Building the focused simulation
- **Boundary inputs:** for every edge entering the subgraph from outside, create a *ghost input*
  with the producer's computed shape. Fill it with real upstream activations (Tier 1: run the
  prefix once and cache it) or with synthetic data (Tier 0).
- **Boundary outputs:** edges leaving the subgraph end in *ghost outputs*.
- Compile a mini SimPlan with `focus` set. The viewport dims everything outside to 10% opacity,
  draws a holographic focus frame, and moves the camera to Focus mode. A **Focus banner**
  ("Simulating: transformer_block (×4), 1 canonical instance") has an Exit button (Esc).

### 7.3 When the selection does not map
If no ref overlaps, or the selection is a function that is never called, or a new snippet typed
in Forge:
1. Offer **"Analyze selection with Fable"** (it costs money; show the estimate).
2. `/api/ai/lens` receives the selection, its surrounding cell, the symbol table and the current
   IR summary, and returns a **snippet-scope Wire IR** (or `pipelines` stages for algorithmic
   code such as `compute_iou`, `generate_anchors` or `encode_box_target`).
3. The snippet is simulated in an isolated **sandbox scene** with ghost inputs. If the snippet is
   a block used by the main model, a link to the main IR lets the user jump there.

### 7.4 Reverse direction
- Click a node or group in 3D or 2D → Monaco reveals its primary ref and adds a neon decoration
  to the lines (all refs get a softer glow, so `declare` and `apply` show up for PyTorch).
- Hover a code line → its nodes pulse in the viewport.

## 8. Later modes (P2)

- **Compare:** two SimPlans on one clock, in a split viewport, with aligned stage markers.
- **Training replay:** parse Keras `fit` logs (`Epoch 3/40 … loss: … accuracy: …`) and PyTorch
  printouts from outputs. Animated curves are synced to an *illustrative* backprop animation
  (reverse pulses whose intensity follows the gradient norm, labelled illustrative). Freeze phases
  are switchable.
- **Receptive field explorer:** click an output pixel and highlight its receptive field back
  through the network onto the input image.
