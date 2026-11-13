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

- [ ] `core/notebook`: nbformat parse (string or list `source`), output stripping, ANSI strip,
      HTML-table-to-text, truncation, stable cell ids.
- [ ] `core/python`: web-tree-sitter + Python grammar (WASM in `public/`), a line-preserving
      sanitizer for `!`, `%` and `%%`, top-level symbol table (assignments, defs, classes, imports).
- [ ] Cell-role heuristics (imports / config / data / **model** / compile / train / eval / viz / other) with confidence.
- [ ] Summary-output parsers: Keras 3 rich table (wrapped rows, `…`, split outputs), Keras 2 plain
      text, torchinfo, "Found N images belonging to K classes", printed param totals.
- [ ] `scripts/make-fixture.ts`. After the user approves, create fixtures for N1–N6 and capture
      raw summary outputs as parser fixtures.
- [ ] UI: drag-drop ingest, Notebook Map, Monaco with all cells (continuous document with cell
      separators), click a cell → scroll to it.
- [ ] IndexedDB project persistence (Dexie) keyed by the notebook hash.

**Acceptance:** all six reference notebooks load in under 1 s after stripping. Model cells are
tagged correctly in all six. The parsed summaries reproduce the totals in doc 07 §4.

---

## Phase 2: IR, Op Catalog, Shape Engine
**Goal:** the deterministic heart, fully tested.

- [ ] `core/ir`: Wire and Runtime Zod schemas (doc 03), normalization (symbol substitution,
      typed params, repeat unrolling, integrity checks), `RepairPatch` apply.
- [ ] `core/ops`: catalog entries for the MVP ops: input, conv2d, depthwise_conv2d, maxpool2d,
      avgpool2d, global_avg_pool, batchnorm, layernorm, relu, gelu, softmax, sigmoid, dense,
      dropout, flatten, reshape, add, concat, multiply, patchify, patch_embed, pos_embed, slice,
      multi_head_attention, embedding, custom, backbone.
- [ ] `core/shapes`: propagation, layout helpers, param/FLOP counting (framework-aware), and
      verification against parsed summaries.
- [ ] Hand-written IR fixtures: AlexNet, VGG16 and mini-ViT (`fixtures/ir/*.json`). These also
      seed the **gallery**.
- [ ] Golden tests: every number in doc 07 §4 for N1, N4 and the N5 models.
- [ ] Fast-path parser (Local-only mode): deterministic IR for simple Keras `Sequential` and
      PyTorch `nn.Sequential` code (AlexNet and VGG16 must parse without AI).

**Acceptance:** `pnpm test` passes all golden facts. The fast path produces a verified IR for
N1's AlexNet and VGG16 with exact totals.

---

## Phase 3: 3D and 2D visualization (static)
**Goal:** a gorgeous, navigable architecture view driven by the IR.

- [ ] `core/layout`: 3D "data river" layout, branch fan-out, group frames, log-scaled tensor boxes, collapse/LOD.
- [ ] Viewport: instanced glyphs per family, curved edges, group frames, labels (drei `Text`/
      `Html` with LOD), bloom, fog, camera modes (Orbit / Overview / Follow placeholder),
      `PerformanceMonitor`.
- [ ] 2D graph view (React Flow + ELK) with a 3D/2D toggle.
- [ ] Inspector (Layer tab): params, shapes with ✓/≈/⚠ badges, param counts, FLOPs, catalog explanation.
- [ ] Code ↔ node linking: click a node → Monaco reveal + decoration; hover a line → node pulse.
- [ ] Model picker for multi-model notebooks. Gallery cards open the prebuilt IR.

**Acceptance:** AlexNet, VGG16 and mini-ViT render at 60 fps. Expand/collapse animates.
Clicking any node highlights the correct code lines. Screenshot regression baselines are recorded.

---

## Phase 4: Fable 5.2 extraction
**Goal:** any notebook → a verified IR through Claude.

- [ ] Load the `claude-api` skill and verify SDK usage. Write `src/server/ai/{config,client,stream,cache,cost-guard}.ts`.
- [ ] System prompt v1 (op catalog and zoo keys generated from code) and the task prompts.
- [ ] Context slicer and condenser (doc 05 §7), plus a token and cost estimate via `countTokens`.
- [ ] `/api/ai/extract` with SSE streaming: Neural Log (thinking summaries), stages, result, typed errors.
- [ ] Refusal and `max_tokens` handling, server-side fallback, usage logging and the cost meter.
- [ ] Verify → repair loop (at most 2 rounds) with `/api/ai/repair` (append-only conversation).
- [ ] Assumptions panel (editable symbols, instant re-verify).
- [ ] Zoo v1 templates: EfficientNetB0, MobileNetV2 (via `tools/zoo-export`, the user runs it
      locally, or hand-authored from the documented architecture with totals checked).
- [ ] First-run privacy notice and Local-only mode switch.
- [ ] `pnpm eval:ai` harness (doc 05 §10). **Ask the user before each paid run.**

**Acceptance:** with the user's approval, the eval on N1–N4 hits every golden total exactly after
verification. N5 extracts all four models. N6 extracts FireNet's structure, with the ConvNeXt-T
backbone collapsed and ✓ on the attention and head params. Re-opening any analyzed notebook makes
no API call.

---

## Phase 5: Simulation engine (MVP complete)
**Goal:** press Play and watch the model think.

- [ ] `core/sim`: SimPlan compiler, pure `stateAt`, clock store, macro-steps for collapsed groups.
- [ ] Viewport choreography: pulses along edges, ignite/transform/emit, flatten ribbon, residual
      merge, frozen ice, repeat ghosts, Follow camera.
- [ ] Timeline: markers by family, group bands, scrub, step, speed, loop, keyboard shortcuts.
- [ ] Tensor backend Tier 1 (tfjs IR interpreter for the MVP ops) and Tier 0 (symbolic), the
      compute budget, and honest labels.
- [ ] Sample inputs plus user image upload. Teaching kernels option for conv1.
- [ ] Micro-visualizers v1: conv-sweep, pool-window, global-pool, activation-curve, norm-histogram,
      dense-matrix, flatten-unroll, dropout-mask, merge-streams, softmax-bars, patchify-grid,
      pos-embed, attention-heads.
- [ ] Inspector Tensor tab (channel grid, histogram, stats).

**Acceptance:** AlexNet, VGG16 and mini-ViT play end to end at 60 fps on a sample image, with
real (random-init) feature maps. Every step's narration and formula show correct numbers.
Scrubbing is deterministic.

---

## Phase 6: Lens
- [ ] `core/lens`: interval index, ref-role priority resolution, subgraph closure, ghost inputs/outputs.
- [ ] Selection chip, focus mode (dim, frame, Focus camera, mini-timeline, `Esc`).
- [ ] `/api/ai/lens` for unmapped selections, plus the sandbox scene.
- [ ] Config-symbol selection → dependent-node highlighting.

**Acceptance:** selecting `transformer_block`'s body in N4 focuses one canonical block (×4) and plays it.
Selecting `PATCH_SIZE = 16` highlights patchify and pos_embed. Selecting N5's `generate_anchors`
(unmapped) produces an anchors pipeline simulation through Fable.

## Phase 7: Forge
- [ ] Palette, editable React Flow graph, live shape validation with fix suggestions, and a stats bar.
- [ ] Codegen (PyTorch and Keras) from catalog templates. Code → IR through the fast path or Fable.
- [ ] `/api/ai/forge` (prompt → IR). Templates (ResNet block, MBConv, Transformer encoder, U-Net level, RPN head, CBAM).
- [ ] Export: `.py`, `.ipynb`, IR JSON. Undo/redo.

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
