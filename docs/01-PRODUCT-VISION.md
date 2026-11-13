# 01 — Product Vision

## The problem

A notebook is a long, linear document. The model architecture, which is the part that matters
most for understanding, is scattered across cells: constants in one cell, a custom `Layer` in
another, a `build_*()` function in a third, a pretrained backbone hidden behind one call such as
`EfficientNetB0(include_top=False)`, and a freeze/unfreeze schedule in a fourth. Reading the code
tells you *what* was written, but not what it *does* to a tensor:

- What does `Conv2D(96, (11, 11), strides=4)` do to a 224×224 image? (It becomes 54×54×96.)
- Why does a mini-ViT's attention layer have 66,368 parameters? (In Keras, `key_dim` is per head,
  so 4 heads × 64 = a 256-wide attention space.)
- What happens inside RoI pooling, anchor generation, or NMS?
- Which layers are frozen in phase 1 of fine-tuning, and which unfreeze in phase 2?

Static diagrams (Netron, `plot_model`) show the graph but not the *process*. Prose explanations
can be wrong. The reference VGG notebook says VGG16 has "3–4x as many parameters" as AlexNet,
but its own summary shows 14.98M against 46.78M.

## The product

**NeuroScope** (working title) is a web app that takes any ML/DL/CV/NLP notebook and turns
its model architecture into a **live, 3D, step-through simulation**. Every simulated piece is
linked to the exact code that created it. It is a microscope for models: built for understanding,
teaching, and designing.

### Primary user

The owner, an ML/CV researcher who builds CNNs, Vision Transformers, multi-task detectors and
classical baselines in Keras and PyTorch (fire detection research, micro-organism classification).
They want to understand, check and explain their models, and to prototype new ones visually.

### Secondary users (later)

Students learning architectures, teachers who need live demos, and reviewers who want to check a
paper's model quickly.

## Goals

| # | Goal |
|---|---|
| G1 | Drop in *any* `.ipynb` and see its model(s) as an accurate, verified 3D architecture within seconds. |
| G2 | Play the forward pass step by step, with a per-layer animation that shows *how* each operation transforms the tensor. |
| G3 | Highlight any part of the code and simulate exactly that part (Lens). |
| G4 | Build new models visually, from code or from plain English, with live validation (Forge). |
| G5 | Correctness you can trust: computed shapes and parameter counts, checked against the notebook's own outputs. |
| G6 | An advanced, futuristic, beautiful interface that makes exploration a pleasure. |

## Non-goals (v1)

- **Not a training platform.** We do not train models or run the user's training loop.
- **No execution of notebook code** (security, heavy environments). We use static analysis plus Fable 5.2.
- Not a general notebook IDE, not a benchmarking tool, no accounts or multi-user collaboration.
- No simulation of data pipelines, augmentation, plots or evaluation code. These cells are shown
  in the Notebook Map as context only.

## Feature set

Priorities: P0 = MVP, P1 = v1, P2 = later.

| ID | Feature | Pri | Summary |
|---|---|---|---|
| F1 | **Notebook ingestion + Notebook Map** | P0 | Drag-drop `.ipynb`. Parse it in the browser, strip images, and show cells as a vertical "map" with role tags (imports, config, data, **model**, compile, train, eval, viz). Model cells glow. |
| F2 | **Architecture extraction** | P0 | Fable 5.2 turns the model cells into the Architecture IR. Multi-model notebooks get a model picker (e.g., AlexNet vs VGG16). Constants are resolved across cells, including values read from outputs (e.g., `num_classes = 8` from "Found 633 images belonging to 8 classes"). |
| F3 | **Verification** | P0 | The shape engine recomputes shapes, params and FLOPs and compares them with `model.summary()` / torchinfo output. Each node gets a badge: ✓ matches summary, ≈ computed, ⚠ mismatch. |
| F4 | **3D viewport** | P0 | Layers as glass volumes sized by tensor shape, glowing edges, collapsible groups (stages, blocks, ×N repeats), and frozen layers rendered as "ice". 2D graph toggle for very large models. |
| F5 | **Forward-pass simulation** | P0 | Timeline with play, pause, step and scrub. Data pulses flow through the graph, the active layer lights up, and the output tensor materializes. Real computation via tfjs on a sample or uploaded input, with random-init weights clearly labelled. |
| F6 | **Inspector + micro-visualizers** | P0 | For the active layer: parameters, in/out shapes, param count, FLOPs, an explanation, and an op-specific animation (kernel sliding for conv, max picking for pooling, attention heatmap, patchify, ...). |
| F7 | **Code ↔ model linking** | P0 | Click a node → the editor jumps to its code. Hover a line → its node glows. |
| F8 | **Lens (selection simulation)** | P1 | Select code → the viewport isolates the matching subgraph and plays only that part, with synthesized inputs of the right shape. If the selection has no match, Fable 5.2 analyzes the snippet on demand. |
| F9 | **AI tutor ("Ask Fable")** | P1 | Ask about the current layer, step or selection, and get answers grounded in the IR and the code. |
| F10 | **Forge (model builder)** | P1 | Build with code, a drag-and-drop graph, or a prompt ("a U-Net for 128×128 segmentation"). Live shape validation, stats, and export to PyTorch, Keras, `.ipynb` or IR JSON. |
| F11 | **Model gallery** | P1 | Pre-analyzed demo models (AlexNet, VGG16, ResNet50, MobileNetV2, EfficientNetB0, ViT, LSTM, Transformer, U-Net, YOLO) that run without an API call. Doubles as onboarding. |
| F12 | **Compare mode** | P2 | Two models side by side with synchronized playback (AlexNet vs VGG16, scratch vs pretrained ViT). |
| F13 | **Training replay** | P2 | Parse recorded `fit()` logs from outputs and replay loss/accuracy curves alongside an illustrative backprop (gradient-flow) animation. Show freeze/unfreeze phases. |
| F14 | **Real weights** | P2 | Optionally load trained weights (ONNX / `.keras`) to show *real* activations of the user's model. |
| F15 | **Export media** | P2 | PNG snapshots and MP4/GIF recordings of a simulation for slides and papers. |

## Coverage roadmap by domain

| Domain | Architectures | Phase |
|---|---|---|
| CV: classification CNNs | AlexNet, VGG, ResNet, Inception/GoogLeNet, MobileNet, EfficientNet, ConvNeXt | MVP → v1 |
| CV: transformers | ViT (scratch + pretrained), patch/position embeddings, CLS token | MVP |
| CV: detection | R-CNN / Fast / Faster R-CNN, RPN, RoI pooling, anchors, NMS, RetinaNet heads, YOLO | v1 |
| CV: segmentation | U-Net, FPN, DeepLab (atrous conv) | v1 → later |
| CV: attention modules | SE, CBAM (channel + spatial), multi-scale fusion | v1 |
| NLP | Embeddings, RNN/LSTM/GRU (unrolled), seq2seq, Transformer encoder/decoder, attention masks | v1 |
| Classical ML | Linear/logistic regression, decision tree, random forest, gradient boosting, SVM, k-NN, k-means, PCA | later |
| Generative | Autoencoder, VAE, GAN, diffusion U-Net | later |
| Audio / time series | 1D conv, temporal conv, RNNs | later |

## Key user journeys

**J1: "Help me understand my ViT notebook."** Drop the file. The Notebook Map lights up the model
cells within about a second (heuristics). The Neural Log streams Fable's reasoning summaries
while the IR is extracted. The 3D viewport assembles: input → Patches → PatchEncoder → 4×
Transformer block (collapsed) → LN → GAP → head. Press **Play**. The image is sliced into 196
patches that fly into a sequence, get projected, and receive position embeddings. Expand a
Transformer block to watch Q/K/V projections and the attention heatmap for each head.

**J2: "What does this block do?" (Lens).** In the ViT notebook, select the body of
`transformer_block()`. The viewport dims everything else and focuses one canonical instance of
the block (marked "×4"). A ghost input of shape (196, 64) feeds it, and only those steps play.

**J3: "Design a new detector" (Forge).** Type "small CNN backbone → RPN head with 9 anchors per
location on a 28×28 map". The graph appears, the shape engine flags a mismatch as you edit, and
you export PyTorch code.

**J4: "Why did my fine-tuning work?"** Open the EfficientNetB0 notebook. The backbone is collapsed
and rendered as frozen ice in phase 1. Switch the phase selector to phase 2: the last N layers
thaw (animated) and the inspector shows the learning-rate change from 1e-3 to 1e-5.

**J5: "Compare AlexNet and VGG16."** (P2) A split view with synced playback, parameter bars
(46.8M vs 15.0M) and receptive-field growth.

## Success metrics

- **Extraction accuracy:** on every golden notebook with a saved summary, computed total params
  match exactly; per-layer output shapes match for ≥ 95% of summary rows (see doc 07).
- **Speed:** Notebook Map in under 1 s. First 3D render from heuristics or the cache in under 2 s.
  Full Fable-verified IR in under 60 s for a typical notebook, with streamed progress the whole time.
- **Fluidity:** 60 fps while playing, with up to 500 visible nodes.
- **Cost:** an analyzed notebook is cached by hash, so re-opening it costs $0. The estimated cost
  is shown before each paid analysis.
- **Understanding:** the user can explain each layer's transformation after one walkthrough
  (self-reported; this is the real goal).
