# 08 — Roadmap

Each phase ends in something demonstrable. Do not start a phase until the previous phase's
acceptance criteria pass. **MVP = Phases 0–5** (Scope works end to end on N1 and N4).
Keep the checkboxes current. They are the project's progress tracker, and CLAUDE.md's *Status*
section points here.

---

## Phase 0: Foundations
**Goal:** an empty but beautiful, correctly structured app.

- [x] Install Node.js LTS and pnpm (the user's machine has neither). `git init`, add `.gitignore`
      (`node_modules`, `.next`, `.env*.local`, `.cache/`, `fixtures/**/raw/`).
- [x] Scaffold Next.js (App Router, TS strict, ESLint, Tailwind v4, `src/` dir) with pnpm.
- [x] Add Vitest and Prettier. The `core/` boundary is an ESLint `no-restricted-imports` rule
      (simpler than `eslint-plugin-boundaries`).
- [ ] Add Playwright (deferred to Phase 3, when there are screenshots worth baselining).
- [x] Design tokens (`src/styles/tokens.css`), fonts via `next/font`, base UI primitives:
      `GlassPanel`, `NeonButton`, `Chip`, `HudReadout`, `Kbd`, `SegmentedControl`.
- [x] App shell: top bar, resizable left/right rails, bottom timeline slot, an empty viewport
      with the animated grid floor (R3F, dynamic import, no SSR).
- [x] Landing page with the drop zone (no logic yet) and the neural-constellation background.
- [x] Update CLAUDE.md's *Commands* and *Status*.

**Status (2026-11-13):** done except Playwright. Both pages were checked in the browser with no
console errors; frame rate was not measured.

**Acceptance:** `pnpm dev` shows the landing page and the Scope shell at 60 fps.
`pnpm lint && pnpm typecheck && pnpm test` pass (one smoke test each). Screenshot check in the browser.

---

## Phase 1: Ingest and Notebook Map
**Goal:** drop any `.ipynb` and see its cells, roles and code instantly, with no AI.

- [x] `core/notebook`: nbformat parse (string or list `source`), output stripping, ANSI strip,
      HTML-table-to-text, truncation, stable cell ids.
- [x] `core/python`: web-tree-sitter + Python grammar (WASM in `public/wasm/`), a line-preserving
      sanitizer for `!`, `%` and `%%`, a symbol table (assignments, defs, classes, imports), and a
      static evaluator for constants (`NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) ** 2` → 196).
- [x] Cell-role heuristics (imports / config / data / **model** / setup / train / eval / plot / other) with confidence.
- [x] Summary-output parsers: Keras 3 rich table (wrapped rows, `…`, split outputs), Keras 2 plain
      text, torchinfo, "Found N images belonging to K classes", printed param totals.
- [x] `pnpm make-fixture` (`scripts/make-fixture.mts`). Fixtures for N1–N6 exist locally in
      `fixtures/notebooks/` and are **git-ignored** (private source code). The raw summary outputs
      in `fixtures/summaries/` are committed as parser fixtures.
- [x] UI: drag-drop ingest (landing page and Scope), file picker, Notebook Map, Monaco with all
      cells as one read-only document, map ⇄ editor selection sync, a notebook inspector.
- [x] IndexedDB project persistence (Dexie) keyed by the notebook hash; the last notebook is
      restored on reload.

**Status (2026-11-13):** done. 77 tests pass. All six notebooks analyze in 35–150 ms, model cells
are tagged exactly, and the parsed summaries reproduce every total in doc 07 §4. Checked in the
browser with the ViT notebook. Not done: Keras 2 and torchinfo parsers are tested on synthetic
samples only (no reference notebook uses them).

**Acceptance:** all six reference notebooks load in under 1 s after stripping. Model cells are
tagged correctly in all six. The parsed summaries reproduce the totals in doc 07 §4.

---

## Phase 2: IR, Op Catalog, Shape Engine
**Goal:** the deterministic heart, fully tested.

- [x] `core/ir`: Wire and Runtime Zod schemas (doc 03), normalization (symbol substitution,
      typed params, repeat unrolling, integrity checks), `RepairPatch` apply.
- [x] `core/ops`: catalog entries for the MVP ops: input, conv2d, depthwise_conv2d, maxpool2d,
      avgpool2d, global_avg_pool, batchnorm, layernorm, relu, gelu, softmax, sigmoid, dense,
      dropout, flatten, reshape, add, concat, multiply, patchify, patch_embed, pos_embed, slice,
      multi_head_attention, embedding, custom, backbone.
- [x] `core/shapes`: propagation, layout helpers, param/FLOP counting (framework-aware), and
      verification against parsed summaries.
- [x] Hand-written IR fixtures: AlexNet, VGG16 and mini-ViT (`fixtures/ir/*.json`). These also
      seed the **gallery**.
- [x] Golden tests: every number in doc 07 §4 for N1, N4 and the N5 models.
- [x] Fast-path parser (Local-only mode): deterministic IR for simple Keras `Sequential` and
      PyTorch `nn.Sequential` code (AlexNet and VGG16 must parse without AI).

**Status (2026-11-14):** done. 123 tests pass. The engine reproduces every N1, N4 and N5 number
in doc 07 §4, row by row against the saved summaries. The fast path reads N1's AlexNet and VGG16
with exact totals and declines the other five notebooks' models (they need the AI step). Scope's
inspector shows the no-AI result. Beyond the roadmap list: `roi_pool` and a `timeDistributed`
param (needed for Fast R-CNN). Not done: zoo expansion (Phase 4), the Lens interval tree (Phase 6;
a sorted `codeIntervals` list exists), `Sequential()` + `.add()` in the fast path, and FLOP counts
have no external reference to check against (only conv and dense follow doc 03's formulas).
The fixtures are generated from `src/core/ir/examples.ts` by `pnpm build-ir-fixtures`.

**Acceptance:** `pnpm test` passes all golden facts. The fast path produces a verified IR for
N1's AlexNet and VGG16 with exact totals.

---

## Phase 3: 3D and 2D visualization (static)
**Goal:** a gorgeous, navigable architecture view driven by the IR.

- [x] `core/layout`: 3D "data river" layout, branch fan-out, group frames, log-scaled tensor boxes, collapse/LOD.
- [x] Viewport: instanced glyphs per family, curved edges, group frames, labels (drei `Text`/
      `Html` with LOD), bloom, fog, camera modes (Orbit / Overview / Follow placeholder),
      `PerformanceMonitor`.
- [x] 2D graph view (React Flow + ELK) with a 3D/2D toggle.
- [x] Inspector (Layer tab): params, shapes with ✓/≈/⚠ badges, param counts, FLOPs, catalog explanation.
- [x] Code ↔ node linking: click a node → Monaco reveal + decoration; hover a line → node pulse.
- [x] Model picker for multi-model notebooks. Gallery cards open the prebuilt IR.

- [ ] Playwright screenshot baselines (still deferred: it needs a browser download of about
      150 MB, and WebGL screenshots are unstable in headless mode; ask the user first).

**Status (2026-11-14):** done except Playwright. 136 tests pass. Checked in the browser: AlexNet,
VGG16 and mini-ViT render in 3D and 2D, groups collapse and expand, clicking a block fills the
Layer inspector and highlights its code line, hovering a code line highlights its block, and the
model picker switches between a notebook's models. Differences from the plan: the 2D view uses
the shared core layout instead of ELK (ADR-014); labels are a DOM overlay positioned per frame,
not drei `Html` (it logged React unmount errors); camera modes are Orbit and Overview only (Follow
needs the simulation, Phase 5); dense layers show at most 18 neurons. Gallery models link to a
generated code listing (`src/core/codegen/listing.ts`) because they have no notebook. Not done:
the frame rate was not measured, frozen/ice and `ai-synth` styling (nothing produces them yet),
and reloading `/scope` with a gallery model open returns to the last notebook.

**Acceptance:** AlexNet, VGG16 and mini-ViT render at 60 fps. Expand/collapse animates.
Clicking any node highlights the correct code lines. Screenshot regression baselines are recorded.

---

## Phase 4: AI extraction (provider-neutral, ADR-013)
**Goal:** any notebook → a verified IR through the user's chosen provider (Gemini, Groq, Qwen, OpenRouter free models; Claude optional).

- [x] One OpenAI-compatible client for Gemini, Groq, Qwen and OpenRouter
      (`src/server/ai/providers.ts`), live model lists, timeouts and retries for overloaded models.
      A provider/model menu in the top bar. (The optional `anthropic` adapter is not built: the
      user has no key.)
- [x] System prompt generated from the op catalog and the zoo, and the Draft format (ADR-015).
- [x] Context slicer (`src/core/ai/context.ts`) with a token estimate shown before sending.
- [x] `/api/ai/extract`: streamed events (stages, model output, result, typed errors) and the
      Analyze panel with a live log.
- [x] Verify → repair loop (at most 2 rounds), inside the same request.
- [x] Assumptions are editable in the inspector, with instant re-verification.
- [x] Zoo **facts** (output shape and parameter counts) for EfficientNetB0, MobileNetV2, ResNet50,
      VGG16, ViT-B/16, ConvNeXt-Tiny. Full layer-by-layer templates are still to do.
- [x] Consent per provider before the first request, and a Local-only switch.
- [x] `pnpm eval:ai <provider> <model> <notebooks…>`. **Ask the user before running it on their notebooks.**
- [ ] Run the eval on the reference notebooks N1–N6 (needs the user's go-ahead: it sends their
      code to a provider).
- [ ] Layer-by-layer zoo templates, so a backbone can be opened and simulated.
- [ ] Usage tracking across runs (the Qwen 1M-token quota is not tracked yet).

**Status (2026-11-15):** built and working end to end. 151 tests pass. Checked with a toy residual
network written for the test: Gemini 2.5 Flash and Groq `gpt-oss-120b` both returned a correct
architecture with code line numbers (58,469 params, hand-checked), the 3D view drew it, and a
reload restored it with no API call. In unit tests, hand-written drafts for the EfficientNetB0 and
MobileNetV2 notebooks reproduce the saved totals exactly (4,214,571 and 2,422,984) through the zoo
facts. **Not done:** no reference notebook has been sent to a provider, so the acceptance below is
untested; Qwen has no key and OpenRouter models were listed but not called; the newest Gemini
models were overloaded or hung during testing (the default is `gemini-2.5-flash`).

**Acceptance:** with the user's approval, the eval on N1–N4 hits every golden total exactly after
verification. N5 extracts all four models. N6 extracts FireNet's structure, with the ConvNeXt-T
backbone collapsed and ✓ on the attention and head params. Re-opening any analyzed notebook makes
no API call.

---

## Phase 5: Simulation engine (MVP complete)
**Goal:** press Play and watch the model think.

- [x] `core/sim`: SimPlan (`buildPlan`), pure `stateAt`, macro-steps for collapsed groups. The
      clock lives in `src/stores/sim-store.ts` outside React state.
- [x] Viewport choreography: pulses along the incoming edges, a glow while the block computes,
      the output picture on each block's face, neuron brightness for dense layers, Follow camera.
- [ ] Still to do in 3D: flatten ribbon, frozen ice, repeat ghosts.
- [x] Timeline: markers by family, scrub, step, speed, loop, keyboard (Space, ←, →, Home).
      Group bands are not drawn.
- [x] Compute: a pure-TypeScript engine in a web worker (ADR-017) for every catalog op, with
      illustrative values for declared layers (pretrained backbones, custom) and anything after
      them. Honest labels in the timeline and the Tensor tab.
- [x] Sample inputs (shapes, stripes, noise) and the user's own picture (read locally).
- [ ] Teaching kernels for the first convolution.
- [x] Visualizers: sliding window (conv, pooling) with the filter's weights, attention maps per
      head, bars for vectors and class probabilities, channel mosaics, token heatmaps.
- [ ] Visualizers still to do: activation curve, norm histogram before/after, dense matrix,
      flatten unroll, dropout mask, merge streams, patch grid, position embedding.
- [x] Inspector Tensor tab (channel grid, histogram, stats).

**Status (2026-11-15):** the core works. 167 tests pass. In the browser, AlexNet ran end to end:
each block showed its feature maps in 3D and the Tensor tab showed the real values. Full runs take
about 0.4 s for the mini-ViT, 2 s for AlexNet and 50 s for VGG16 (measured in Node; layers appear
as they finish, and playback waits for them). **Not verified:** continuous playback and frame rate.
The preview pane was hidden during testing and drew no animation frames, so only single steps and
still frames were checked. Pulses, the glow and Follow mode are therefore untested on screen.

**Acceptance:** AlexNet, VGG16 and mini-ViT play end to end at 60 fps on a sample image, with
real (random-init) feature maps. Every step's narration and formula show correct numbers.
Scrubbing is deterministic.

---

## Phase 6: Lens
- [x] `core/lens`: resolve a code selection to a focus by priority (a group's definition or loop,
      then the layers the lines create, then constants), close gaps between selected layers, and
      find the layers feeding and reading the focus. It scans the IR's code refs directly; an
      interval tree is not needed at this size.
- [x] Selection chip in the Code panel header ("Simulate selection", shortcut `F`), focus mode
      (everything else dimmed, a frame around the focus, the camera moves to it, the timeline
      steps only through the focused layers, `Esc` or Exit leaves).
- [x] Selecting a constant highlights the layers that depend on it (`symbolsUsed` on each node).
- [ ] `/api/ai/lens` for selections that define no layer of the model, plus the sandbox scene.
- [ ] Ghost input and output markers at the edge of the focus (the layers are known, not drawn).

**Status (2026-11-15):** done except the AI path for unmapped code. 181 tests pass. Checked in
the browser on the mini-ViT gallery model: selecting one Transformer block's lines focused its
9 layers, dimmed the rest, framed it and limited the timeline to 9 steps. The unit tests cover
the acceptance cases below on the example IR (the block's definition gives one copy of 4;
`PATCH_SIZE` gives the patch layer). **Not verified:** the same on the real ViT notebook, which
needs an AI result for it first. Constants only link to layers when the IR was written with
symbol names, which the AI draft and the gallery models do and the no-AI reader does not.

**Acceptance:** selecting `transformer_block`'s body in N4 focuses one canonical block (×4) and plays it.
Selecting `PATCH_SIZE = 16` highlights patchify and pos_embed. Selecting N5's `generate_anchors`
(unmapped) produces an anchors pipeline simulation through the AI step.

## Phase 7: Forge
- [x] Palette (20 layers, click or drag onto the view), templates (small CNN, residual block, MLP,
      Transformer encoder), a settings form per layer, live shape checking with a plain-language
      fix for each common error, totals and a problem list.
- [x] The 2D graph is editable in Forge: drag between cards to connect, double-click a line to
      remove it. Layout stays automatic (cards cannot be placed by hand).
- [x] Export of runnable code: Keras functional API or a PyTorch `nn.Module`
      (`src/core/codegen/export.ts`), as `.py`, `.ipynb` or IR JSON. Undo and redo.
- [x] "Describe a model": plain English → a model, through the AI step (`mode: "design"`).
- [ ] Code → model inside Forge (paste code). Scope already does this for notebooks.
- [ ] More templates (MBConv, U-Net level, RPN head, CBAM), repeat groups and grouping in Forge.

**Status (2026-11-15):** built. 200 tests pass. In the browser: the page opened with the small
CNN, adding a convolution after the classifier was caught as a shape error with a suggested fix,
a setting change updated the label, undo restored the model, and switching to PyTorch rewrote
the code. The Forge model simulates like any other. Groq designed a valid CNN from a one-line
description in 4 s. **Not verified:** the exported scripts pass Python's syntax check, but they
were not run, because neither PyTorch nor Keras is installed on this machine. Connecting cards
by dragging in the 2D view and dragging from the palette were not tried in the browser.

**Acceptance:** build a small CNN by drag-and-drop, see a shape error caught, export runnable PyTorch code, and simulate it.

## Phase 8: Breadth
- [ ] Zoo: ResNet50, VGG16, ConvNeXt-T, ViT-B/16 (KerasHub), YOLOv8n, torchvision ResNet50.
- [ ] Detection: anchor_generator, roi_pool/roi_align, box_decode, nms ops and visualizers; pipeline stages.
- [ ] Attention modules: channel/spatial attention, squeeze-excite (FireNet ✓).
- [ ] NLP: embedding, rnn/lstm/gru (recurrent-unroll), Transformer encoder/decoder with masks, a toy tokenizer.
- [ ] Classical ML (`estimator` + visualizers): decision tree, random forest, logistic regression, k-means, SVM, PCA.

## Phase 9: Polish and P2 features
- [ ] Compare mode, training replay from logs, receptive-field explorer.
- [ ] Real weights (ONNX / `.keras`) → Tier 2.
- [ ] Export PNG/MP4, onboarding tour, light theme, colorblind palette, sound.
- [ ] Performance pass (workers, texture budgets), an accessibility audit (the `design:accessibility-review` skill).

---

## First session checklist (next time)
1. Read CLAUDE.md, this file and doc 02.
2. Confirm with the user: the product name, whether to install Node LTS + pnpm (and how: winget
   or the installer), and permission to create fixtures from their notebooks.
3. Do Phase 0.
