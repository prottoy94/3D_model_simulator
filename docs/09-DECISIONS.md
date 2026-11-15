# 09 — Decisions (ADR log) and Open Questions

Format: context → decision → consequences. Add new ADRs at the bottom and never rewrite old
ones. Supersede them with a new ADR instead.

---

### ADR-001: One TypeScript / Next.js app
- **Context:** we need a rich 3D client, a safe place for the API key, and no Python execution in v1.
- **Decision:** a single Next.js (App Router) app in TypeScript. Server route handlers call the
  Anthropic TS SDK. No separate backend, no monorepo. Module boundaries (`src/core` pure) are
  enforced by lint.
- **Consequences:** one language and one toolchain. Python static analysis uses web-tree-sitter
  in the browser. If we ever need real Python (a tracer, weight conversion), it is an optional
  local sidecar, not core.

### ADR-002: Never execute user notebook code
- **Context:** notebooks import heavy frameworks, read private data, and can run arbitrary commands.
- **Decision:** v1 is static analysis plus LLM only. The only Python we ever run is our own
  `tools/zoo-export` scripts, run manually by the developer.
- **Consequences:** shapes come from the shape engine, not from tracing. Accuracy relies on the
  catalog, the zoo and verification against saved outputs. A future opt-in local tracer
  (torch.fx / `model.to_json()` in an isolated venv, model cells only, explicit consent) stays
  possible but is out of scope.

### ADR-003: A flat graph IR with groups via `parentId`
- **Context:** Claude structured outputs reject recursive schemas and need `additionalProperties: false` everywhere.
- **Decision:** flat `nodes`, `edges` and `groups` arrays. Hierarchy uses `parentId`. Params are `{key, value}[]` on the wire.
- **Consequences:** easy diffing and patching. Normalization turns params into typed objects per
  op. The runtime IR adds indexes for fast traversal.

### ADR-004: The LLM proposes, the engine verifies
- **Context:** LLMs can be wrong about shapes and counts, and so can notebook prose (N1's VGG claim).
- **Decision:** all numbers shown to the user come from the deterministic shape engine and the op
  catalog, cross-checked against saved summaries. LLM claims only trigger repairs.
- **Consequences:** every op needs exact shape and param functions with tests. Trust badges
  (✓/≈/⚠) appear throughout the UI.

### ADR-005: Claude Fable 5.2 as the AI engine
- **Context:** the user asked for Fable 5.2.
- **Decision:** `claude-fable-5-2` for every AI task, with effort tuned per task (low → high).
  Server-side refusal fallback is enabled (`fallbacks: "default"`). The model id and effort table
  live in one config file.
- **Consequences:** $8/$40 per MTok, so aggressive caching, context slicing and cost estimates
  are required. 30-day retention is required (no ZDR), so there is a privacy notice and a
  Local-only mode. Switching a task to a cheaper model is the user's call. The config makes it a
  one-line change, but we do not do it without being asked.

### ADR-006: React Three Fiber for 3D, React Flow + ELK for 2D
- **Decision:** R3F + drei + postprocessing for the viewport. React Flow + elkjs for the 2D and Forge editor.
- **Consequences:** a React-native mental model. Per-frame animation must use refs and
  `useFrame` (no React state per frame).

### ADR-007: tfjs for in-browser compute, with a symbolic fallback
- **Context:** the simulation is more convincing with real numbers, and we cannot run Python.
- **Decision:** an IR interpreter built on tfjs ops (WebGPU/WebGL), with random-init weights
  (Tier 1). Symbolic procedural activations (Tier 0) for large or unsupported models.
- **Consequences:** label "random-init" everywhere. **Re-check tfjs maintenance status at
  scaffold time.** If it has stagnated, use a small set of custom WebGPU kernels for the MVP ops
  (conv, matmul, pool, norm, softmax) behind the same `TensorBackend` interface.

### ADR-008: Monaco for code
- **Decision:** `@monaco-editor/react`, with decorations for ref highlights and the selection API for Lens.
- **Consequences:** a large bundle, so it is lazy-loaded. CodeMirror 6 is the fallback if bundle size becomes a problem.

### ADR-009: Local-first, no accounts in v1
- **Decision:** projects and caches live in IndexedDB. The server keeps only the AI result cache on disk.
- **Consequences:** zero auth work. Sharing and collaboration are future features.

### ADR-010: Model zoo templates for pretrained backbones
- **Context:** `EfficientNetB0(...)`, `from_preset(...)`, `convnext_tiny(...)` and `YOLO(...)` hide hundreds of layers.
- **Decision:** curated, versioned IR templates (exported from the real libraries by developer
  scripts, totals checked). AI-synthesized templates are allowed for unknown backbones but flagged.
- **Consequences:** exact backbone params (e.g., EfficientNetB0 + head = 4,214,571), and the
  backbone is collapsed by default for readability.

### ADR-011: Notebook fixtures stay local; only summary tables are committed
- **Context:** the reference notebooks hold the user's research code, and the repo is on GitHub.
- **Decision:** `fixtures/notebooks/` is git-ignored. Tests that need a notebook skip when it is
  missing. `fixtures/summaries/` (layer tables only) is committed.
- **Consequences:** a fresh clone runs fewer tests until fixtures are recreated with
  `pnpm make-fixture`. To publish a fixture on purpose, add a `.gitignore` exception.

### ADR-012: Monaco loads from its default CDN for now
- **Context:** `@monaco-editor/react` fetches the editor from jsDelivr by default. Bundling it
  locally needs worker configuration under Turbopack.
- **Decision:** keep the CDN default in Phase 1. Only editor code is fetched; notebook content is
  never sent anywhere.
- **Consequences:** the code panel needs internet on first load. Revisit (self-host via
  `loader.config`) before any offline or deployed use.

### ADR-013: Provider-neutral AI layer (supersedes ADR-005)
- **Context:** the user has no Claude API key. They have free-tier keys for Gemini, Groq and
  OpenRouter, and a Qwen (Alibaba Model Studio) free quota of 1M tokens.
- **Decision:** one `AiProvider` interface with an adapter per provider (`gemini`, `groq`, `qwen`,
  `openrouter`, optional `anthropic`). The user chooses the provider and model in the app; only
  providers with a key in `.env.local` are offered; model lists are loaded live. OpenRouter is
  limited to free models. Gemini is the default. Details in doc 05 §0.
- **Consequences:** prompts and schemas cannot rely on Claude-only features. Structured output is
  best effort and always re-validated with Zod. Weaker models mean more repair rounds, which makes
  the deterministic engine and the no-AI fast path (Phase 2) more important. Free tiers may train
  on prompts, so the privacy notice names the provider.

### ADR-014: One layout core for 3D and 2D; ELK postponed (amends ADR-006)
- **Context:** ADR-006 planned React Flow + ELK for the 2D graph. Phase 3 needed the same
  structure in 3D and 2D, including collapsed groups, and model graphs here are mostly chains
  with a few branches and skip connections.
- **Decision:** `src/core/layout/graph.ts` builds one "visible graph" (collapsed groups folded,
  ranks by longest path, lanes for branches). The 3D river and the 2D cards are two placements of
  it. React Flow renders the 2D view; ELK is not installed.
- **Consequences:** both views always agree, the layout is pure and unit-tested, and there is no
  1.5 MB layout dependency. Dense graphs with many crossing edges (large detectors) may need ELK
  later; it can replace the lane step without touching the views.

### ADR-015: The LLM writes a forgiving "Draft", not Wire IR
- **Context:** Wire IR (doc 03) is strict: every field required, edges separate from nodes, params
  as key/value pairs. Free-tier models are weaker than the model the plan was written for, and
  only some providers enforce a JSON schema.
- **Decision:** the model writes a Draft (`src/core/ai/draft.ts`): few required fields, params as
  a plain object, connections as an `inputs` list on each node, code location as
  `[cell, first, last]`. `draftToWire` expands it into Wire IR; unknown layer names fall back to
  `custom`, arithmetic like `"num_classes + 1"` is evaluated by a tiny parser, and pretrained
  backbones are filled in from zoo facts. Repairs are partial node patches, not `RepairPatch`.
- **Consequences:** Wire IR stays the internal contract and the only thing the engine reads.
  Requests use plain JSON mode and are validated with Zod. `RepairPatch` (doc 03 §6) remains for
  Forge and for a future strict-schema provider.

### ADR-016: One OpenAI-compatible client instead of one adapter per provider
- **Context:** ADR-013 planned an adapter per provider. All four configured providers expose an
  OpenAI-compatible chat endpoint (checked against the live APIs on 2026-11-15).
- **Decision:** one streaming client with a small config per provider (address, key, which models
  to offer, a data note). Model lists are loaded live and filtered; OpenRouter is limited to
  `:free` ids with a zero price.
- **Consequences:** adding a provider is a config entry. Provider-specific features (Gemini
  schema-constrained output, Qwen thinking mode) are not used. An `anthropic` adapter would be
  the first real second client.

### ADR-017: A pure-TypeScript compute engine instead of tfjs (replaces ADR-007's choice)
- **Context:** ADR-007 chose tfjs for in-browser compute and allowed custom kernels behind the
  same interface if tfjs was a poor fit. The simulation needs every op in the catalog, exact
  agreement with the shape engine, and deterministic numbers for tests.
- **Decision:** `src/core/sim` computes each layer on `Float32Array`s in plain TypeScript, in a
  web worker. Weights are generated from a seed per node (Glorot uniform, as an untrained Keras
  layer); a dense layer's weight rows are generated on the fly, so AlexNet's 26-million-weight
  layer needs no memory. Tensors are channel-last internally whatever the framework.
- **Consequences:** no dependency, the engine is unit-tested in Node, and results are identical
  on every machine. It is slow for big CNNs: about 0.3 billion multiply-adds per second, so VGG16
  takes about 50 s (layers appear as they finish). A WebGPU or tfjs backend can replace the
  kernels later without touching the plan, the views or the UI. Trained weights are not loaded.

---

## Open questions (ask the user when relevant)

| # | Question | Default if unanswered |
|---|---|---|
| Q1 | Product name? ("NeuroScope" is a working title.) | Keep NeuroScope |
| Q2 | Local-only use, or deploy (e.g., Vercel)? Deploying affects request time limits for long Fable calls and the need for auth. | Local first, deploy-ready structure |
| Q3 | AI budget per month? | `AI_MONTHLY_BUDGET_USD=50`, with an estimate shown before each run |
| Q4 | ~~Copy stripped notebooks into `fixtures/`?~~ Resolved by ADR-011: local only, git-ignored | n/a |
| Q5 | Priority after the MVP: detection (N5/N6), NLP, or classical ML? | Detection first (it matches the user's research) |
| Q6 | Should Forge default to PyTorch or Keras codegen? | Keras for the micro-organism work, PyTorch for research. Make it a toggle, default PyTorch |
| Q7 | Load real trained weights (Tier 2) early? | P2 |
| Q8 | Use a cheaper model for the tutor (`explain`) to save cost? | No. Fable 5.2 everywhere unless the user says otherwise |
