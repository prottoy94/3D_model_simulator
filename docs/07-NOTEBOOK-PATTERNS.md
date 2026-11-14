# 07 — Notebook Patterns and Golden Test Facts

These lessons come from the six reference notebooks the user provided. We use them **for ideas
and as test fixtures only**: the product must handle *any* notebook, and these six cover a useful
spread of the patterns real notebooks use.

| # | Notebook | Framework | What it defines |
|---|---|---|---|
| N1 | `Micro-organism Classificationn/microorganism_classification_alexnet_vgg_scratch.ipynb` | Keras (TF) | AlexNet (Sequential, + BatchNorm) and VGG16 (Sequential, GAP head), from scratch |
| N2 | `…/microorganism_classification_efficientnet_finetuned.ipynb` | Keras | `keras.applications.EfficientNetB0(include_top=False)` + head, 2-phase fine-tuning |
| N3 | `…/microorganism_classification_mobilenetv2.ipynb` | Keras | `MobileNetV2(include_top=False)` frozen + head |
| N4 | `…/microorganism_classification_vit_scratch_and_pretrained.ipynb` | Keras + KerasHub | Mini-ViT from scratch (custom `Layer`s, functional API, loop of blocks), and pretrained ViT-B/16 via `keras_hub.ViTBackbone.from_preset` + CLS-token head, 2-phase fine-tuning by blocks |
| N5 | `…/microorganism_object_detection_rcnn_family.ipynb` | Keras | R-CNN (two CNNs + selective search + NMS), Fast R-CNN (custom `ROIPoolingLayer`, `TimeDistributed`, 2 inputs / 2 outputs), Faster R-CNN (anchors, RPN), on synthetic scenes |
| N6 | `D:\Research\Fire detection\fire_research_final_v3.ipynb` | PyTorch + torchvision + Ultralytics | `FireNet`: ConvNeXt-Tiny backbone → CBAM-style multi-scale attention (P3/P4/P5) → classification head + RetinaNet-style detection head. Baselines: ResNet50 classifier and YOLOv8n |

## 1. File-format realities

- **nbformat 4.5**, kernels `tfenv` / Python. `source` can be a list of strings or a single
  string. Handle both.
- **Size comes from images:** the files are 3.5–9.9 MB, mostly base64 `image/png` outputs. Code
  and markdown are only 16–190 KB. **Strip images in the browser before anything else.**
- **Output types:** `stream`, `execute_result` and `display_data` with `text/plain`, `text/html`,
  `text/markdown`, `image/png`, and `application/vnd.jupyter.widget-view+json`. Keep only text.
- **Keras 3 `model.summary()` is rich text:** a box-drawing table (`┏━┳┓ │ ├─┼┤`) with **ANSI
  color codes** (`\x1b[1m`, `\x1b[38;5;…m`). Layer names wrap onto a second row
  (`batch_normalization` / `(BatchNormalization)`) or get truncated with `…`
  (`multi_head_attenti…`). The table is **split across several `display_data` outputs** (title,
  table, "Total params", "Trainable params", "Non-trainable params"). The parser must strip ANSI,
  join outputs, merge wrapped rows, and treat `…` names as prefixes.
- A Keras 3 functional summary has a 4th column, **"Connected to"**, which gives edge hints.
- Colab specifics: `from google.colab import drive; drive.mount(...)` and Drive paths. Treat them
  as data-loading noise.
- Shell and magic lines: `!pip install --quiet keras-hub`. Sanitize them to comments **in place**
  so line numbers survive.

## 2. Architecture-definition patterns to support

| Pattern | Example (notebook) | IR handling |
|---|---|---|
| Keras `Sequential([...])` inside a `build_*()` function | N1 AlexNet, VGG16 | Linear chain. Fused `activation=` on Conv2D/Dense. Parameters come from function args resolved at the **call site** (`build_alexnet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=num_classes)`). |
| Keras functional API (`Input` → layers → `Model(inputs, outputs)`) | N4 mini-ViT, N5 | Explicit graph. Multiple inputs and outputs (`Model([image_input, roi_input], [cls, bbox])`). |
| Custom `Layer` subclasses (`call()` with `tf.*` ops) | N4 `Patches`, `PatchEncoder`; N5 `ROIPoolingLayer` | Map to catalog ops where they match (`patchify`, `patch_embed` + `pos_embed`, `roi_pool`), or use `custom` with a declared shape. `PatchEncoder` = Dense projection + learned position `Embedding` added. |
| Python helper that builds a block | N4 `transformer_block(x, ...)` | A `definition` ref on the function body, `call` refs per call site, a repeat group. |
| Python loops that repeat blocks | N4 `for _ in range(TRANSFORMER_LAYERS)` | A `repeat` group with `repeatSymbol` and a `loop` ref. Unroll at normalization. |
| Helper reused for two separate models | N5 `build_small_cnn_backbone()` used for the classifier and the regressor | Two models, each with its own instance of the helper's nodes, both refs pointing at the same definition lines (Lens lights up both). |
| Pretrained backbone + new head | N2 EfficientNetB0, N3 MobileNetV2: `base_model.output → GAP → Dense(128) → Dropout → Dense(C)` | A `backbone` group (`zooKey`) + head nodes. `include_top=False` drops the classifier in the template. |
| Preset-based pretrained model | N4 `keras_hub.models.ViTBackbone.from_preset("vit_base_patch16_224_imagenet")` | zooKey `keras_hub.ViTBackbone:vit_base_patch16_224_imagenet`. Output `(B, 197, 768)`; `Lambda(lambda t: t[:, 0, :])` = a `slice` node (CLS token). |
| Freeze / fine-tune phases | N2: `base_model.trainable=False`, then unfreeze the last `FINE_TUNE_LAYERS = 30` layers at LR 1e-5. N4: unfreeze the last `FINE_TUNE_BLOCKS = 2` of 12 encoder blocks; patch embedding stays frozen | `training.phases` with `frozenGroupIds` and `unfreezeRule`. The viewport phase selector toggles the ice rendering. |
| `TimeDistributed(Dense(...))` over RoIs | N5 Fast R-CNN | A `time_distributed` wrapper → the inner op applied per RoI. Shape `(B, R, …)` where `R` is a symbolic dim. |
| `Reshape` to a flat anchor list | N5 RPN `Reshape((TOTAL_ANCHORS, 1))` | A `reshape` node. TOTAL_ANCHORS = 28·28·9 = 7,056, resolved via symbols. |
| PyTorch `nn.Module` hierarchy | N6 `FireNet` → `MultiScaleAttention` → `CascadedAttention` → `ChannelAttention` / `SpatialAttention` | Groups per module. `declare` refs in `__init__`, `apply` refs in `forward`. |
| `nn.Sequential`, `nn.ModuleList`, list comprehensions | N6 `self.adapt = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in stage_channels])`, `[a(m(f)) for f, m, a in zip(...)]` | Expand per element (3 levels: P3/P4/P5) into parallel branches. |
| Functional tensor ops in `forward` | N6 `F.adaptive_avg_pool2d`, `torch.cat`, `F.interpolate`, `x.mean(1)`, `x.amax(1)`, `x * torch.sigmoid(a)` | Each becomes a node (`global_avg_pool`, `concat`, `upsample`, `reduce_mean`, `reduce_max`, `multiply`, `sigmoid`). |
| Slicing pretrained features into stages | N6 `feats = convnext_tiny(...).features; self.stem, self.s1, self.d2, … = (feats[i] for i in range(8))` | A zoo template for `torchvision.models.convnext_tiny` with stage ids, plus the taps f2/f3/f4 → P3/P4/P5. |
| Multi-task heads with one shared encoder | N6 `forward(x, tasks=("cls","det"))`, `classify()` and `detect()` | One model with two output nodes (cls logits, det cls + box). `train_forward` is recorded as a variant. |
| Third-party opaque detectors | N6 `YOLO("yolov8n.pt")` | zooKey `ultralytics.yolov8n`, a collapsed backbone/neck/head template. |
| Non-NN architectural algorithms | N5 selective search, IoU, NMS, `encode_box_target`; N5/N6 anchor generators, delta encode/decode | `pipelines` stages with an `AlgorithmKind`, plus detection ops where they sit inside the model. |
| Config objects and run modes | N6 `@dataclass Config(img_size=224, …)`; a `RUN_MODE == "smoke"` branch sets `img_size=96` | Symbol resolution must follow the active branch. Prefer printed evidence ("anchors per image at 224px: 9,261") and record it as `from-output`. |
| Values only known at runtime | `num_classes = len(class_names)` | Resolve from outputs: "Number of micro-organism types (classes) found: 8", "Found 633 images belonging to 8 classes." Provenance `from-output`, editable in the Assumptions panel. |

## 3. Things that trip naive approaches

- **Prose can be wrong.** N1's markdown says VGG16 has "roughly 3–4x as many parameters as our
  AlexNet build", but the summaries show 14.98M vs 46.78M (the GAP head removes the giant dense
  layers). Always trust computed numbers and summaries over markdown.
- **Keras `key_dim` is per head.** N4 `MultiHeadAttention(num_heads=4, key_dim=PROJECTION_DIM=64)`
  makes a 256-wide attention space, which gives 66,368 params rather than the ~16k you would
  expect from 64/4 per head. The simulator should *teach* this.
- **Static shapes Keras could not infer.** The summary shows `patches (Patches) → (None, None, 768)`.
  Our engine computes 196 tokens. Treat it as a match (see doc 03 §6).
- **Keras vs PyTorch param accounting:** Keras counts BatchNorm moving statistics as
  non-trainable params (AlexNet: 704 = 2·96 + 2·256). PyTorch does not count buffers.
- **NHWC (Keras) vs NCHW (PyTorch)** in one product: display in native layout and compute through layout helpers.
- **Two forward paths** (N6 `forward` vs `train_forward`) and **conditional heads**
  (`heads=("cls","det")`, `with_fuse=("cls" in heads)`).
- **The same lines create several nodes** (helper functions, loops, comprehensions): Lens must
  map one range to many nodes.
- **Markdown cells explain the architecture well** (e.g., N6 cell 8 gives the stage output sizes
  28×28×192 / 14×14×384 / 7×7×768). Send the nearby markdown to Fable, but verify the numbers.

## 4. Golden facts (permanent tests)

All values come from the notebooks' own saved outputs (`model.summary()` / prints), extracted on
2026-11-13. "Computed" means hand-verified with the formulas in doc 03 §4.3 and matching the output.

### N1: AlexNet (input 224×224×3, 8 classes)
| Check | Expected |
|---|---|
| conv1 output / params | `(None, 54, 54, 96)` / 34,944 |
| bn1 params | 384 |
| pool1 output | `(None, 26, 26, 96)` |
| conv2 output / params | `(None, 26, 26, 256)` / 614,656 |
| pool2 output | `(None, 12, 12, 256)` |
| conv3 / conv4 params | 885,120 / 1,327,488 |
| flatten size (computed) | 5·5·256 = 6,400 |
| fc1 params (computed) | 26,218,496 |
| **Total / trainable / non-trainable** | **46,781,192 / 46,780,488 / 704** |

### N1: VGG16 variant (13 conv, GAP, Dense 512, Dense 8)
- Total / trainable / non-trainable: **14,981,448 / 14,981,448 / 0**
  (conv stack 14,714,688 + Dense 512 262,656 + Dense 8 4,104)

### N2: EfficientNetB0 (`include_top=False`, imagenet) + GAP → Dense 128 → Dropout 0.3 → Dense 8
| Check | Expected |
|---|---|
| Total params | **4,214,571** |
| Phase 1 trainable (head only) | **165,000** (1280·128+128 + 128·8+8) |
| Phase 2 trainable (last 30 layers unfrozen) | **1,661,160** |
| Phase 2 non-trainable | 2,553,411 |

### N3: MobileNetV2 (frozen) + the same head
- Total / trainable / non-trainable: **2,422,984 / 165,000 / 2,257,984**
  (the backbone also ends in 1280 channels, so the head has the same 165,000 params as N2)

### N4: Mini-ViT from scratch (patch 16, D 64, 4 heads, 4 blocks, MLP 128)
| Check | Expected |
|---|---|
| tokens | 196 (14×14), patch vector 768 |
| PatchEncoder params | 61,760 (768·64+64 + 196·64) |
| LayerNormalization params | 128 |
| MultiHeadAttention params | **66,368** |
| **Total** | **404,040** (all trainable) |
| Pretrained ViT-B/16 output | `(None, 197, 768)`, 12 encoder blocks, last 2 fine-tuned in phase 2 |

### N5: R-CNN family (canvas 224, RoI crops 64, 8 classes + background)
| Model | Total params (all trainable) | Notes |
|---|---|---|
| `RCNN_Classifier` | **1,143,113** | 3× (Conv+Pool) on 64×64 → Flatten 8192 → Dense 128 → Dense 9 |
| `RCNN_BBox_Regressor` | (same backbone, Dense 4 head) | computed: 1,142,468 |
| `FastRCNN` | **897,869** | shared 28×28×128 map → RoIPool 7×7 → TimeDistributed Dense 128 → cls 9 / bbox 4 |
| `RPN` | **246,637** | rpn_conv 3×3×128 → objectness 1×1×9 (sigmoid) + bbox 1×1×36 → Reshape (7056,1)/(7056,4) |
| Anchors | 28·28·9 = **7,056** | scales [32, 64, 128] × ratios [0.5, 1, 2], stride 8 |

### N6: FireNet (PyTorch, 224 px)
| Check | Expected |
|---|---|
| Total params | **32.18 M** (printed) |
| ConvNeXt-T stage taps | 28×28×192, 14×14×384, 7×7×768 (markdown; verify via the zoo template) |
| Attention adapters | 1×1 conv → 256 channels per level, CBAM (channel reduction 16, spatial k=7) |
| Anchors per image | **9,261** = (28² + 14² + 7²) · 9 |
| Detection head | 2 towers × 3 (conv3×3 + GroupNorm(32) + ReLU), cls_out 9·2, box_out 9·4, shared across levels |
| Classification head | concat(avg, max) 512 → Dropout 0.3 → Linear 256 → GELU → Dropout 0.2 → Linear 2 |
| Baselines | ResNet50 (fc → 2), YOLOv8n (Ultralytics) |

Computed check for the regressor: the backbone (896 + 18,496 + 73,856) + Dense 128 (1,048,704)
+ Dense 4 (516) = 1,142,468.

## 5. Fixtures (as built in Phase 1)

- `pnpm make-fixture <notebook.ipynb> <short-name>` strips image and widget outputs, keeps code,
  markdown and text outputs, and writes `fixtures/notebooks/<short-name>.ipynb`. These copies are
  **git-ignored**: they contain the notebooks' full source, which may be unpublished research.
  Tests that need them skip themselves when a file is missing.
- Short names: `n1-alexnet-vgg`, `n2-efficientnet`, `n3-mobilenetv2`, `n4-vit`, `n5-rcnn`, `n6-firenet`.
- `fixtures/summaries/*.txt` holds the raw summary outputs from N1–N5 (layer tables only). These
  are committed and drive the summary-parser tests.
- The expectations (model cells, summary totals, constants, class counts) live in
  `src/core/notebook/reference.test.ts`.
- Model cells found by the heuristics: N1 [14, 26], N2 [16, 17], N3 [15, 16], N4 [16, 40, 41],
  N5 [26, 38, 43, 58, 66] (cell 66 builds the Faster R-CNN detector head), N6 [9].
- `pnpm tsx scripts/inspect-notebook.mts <file>` prints what the analysis sees in any notebook.
