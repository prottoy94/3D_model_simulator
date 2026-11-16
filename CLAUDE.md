# CLAUDE.md — NeuroScope (working title)

> A futuristic, 3D, live simulator that turns any ML / DL / CV / NLP Jupyter notebook (`.ipynb`)
> into an explorable, animated model of its **architecture**, so the person who wrote it can *see*
> how it works. It also lets them design new models and watch them run.

## Status

- **Phases 0–7 are built; Phases 8 and 9 are partly built (2026-11-16). Next: the open boxes
  of Phase 8 (classical ML, detection post-processing, NLP decoder) and Phase 9 (real weights,
  training replay, MP4, performance and accessibility passes)** in
  [docs/08-ROADMAP.md](docs/08-ROADMAP.md). Phase 9 so far: receptive fields
  (`src/core/receptive`, shown in the Layer inspector), Compare (`src/core/compare`,
  `src/features/compare`), the Display menu with a light theme and a colorblind-safe palette
  (`usePrefsStore`; token overrides in `tokens.css`), "Save picture" (PNG of the 3D view) and a
  guided tour (`src/features/shell/tour.tsx`, targets are `data-tour` attributes). Phase 8 so far: 17 more ops in `src/core/ops/extra.ts`
  with kernels in `src/core/sim/kernels-extra.ts`, shared weights (`sharedWith`), backbone stage
  taps, and FireNet as a gallery model (`src/core/gallery.ts`, the gallery's model list).
  Phase 7 added Forge: `src/core/forge` (pure edits on a Draft, templates, fix suggestions),
  `src/core/codegen/export.ts` (runnable Keras and PyTorch), `useForgeStore` (document, undo,
  local save) and `src/features/forge`. Forge shows its model through `useModelStore` with
  source `"forge"`, so the 3D/2D views, the simulation and the inspector are reused as they are.
  Phase 6 added Lens: `src/core/lens` maps selected code to a focus, `useModelStore.focus` holds
  it, the views dim everything outside it, and the sim plan is filtered to it (`focusPlan`).
  Not built: AI analysis of code that defines no layer.
  Open items: the AI eval on the reference notebooks (needs the user's go-ahead) and checking
  continuous playback on screen. Phase 5 added the simulation: `src/core/sim` (kernels, runner,
  views, plan; ADR-017), a web worker (`src/features/sim/sim.worker.ts`), `useSimStore` with the
  clock in `simClock`, the timeline bar, `sim-layer.tsx` in the 3D view and the Tensor tab.
  Phase 4 added the AI step: `src/core/ai` (Draft format, prompt, context slicer, problem
  reports), `src/core/zoo` (backbone facts), `src/server/ai` (providers, verify → repair loop),
  `/api/ai/providers` and `/api/ai/extract`, and `src/features/ai` (menu, Analyze panel).
  AI results are saved per notebook in IndexedDB. Phase 3 added the layout core (`src/core/layout`), the 3D
  model view, the 2D graph (React Flow), the Layer inspector, two-way code ↔ model linking, the
  model picker and three gallery models. `useModelStore` holds the IR on screen, the selection
  and the collapsed groups; every view reads it. Phase 2 added the Wire/Runtime IR (`src/core/ir`), the
  op catalog (`src/core/ops`), the shape engine and summary verification (`src/core/shapes`) and
  the no-AI reader for simple layer stacks (`src/core/fastpath`). `buildRuntimeIR(wire, { summaries })`
  is the one entry point from Wire IR to everything a view needs. What works now: drop a notebook → it is parsed and
  stripped in the browser, every cell gets a role, constants are resolved, saved `model.summary()`
  outputs are parsed, and Scope shows the Notebook Map, the code (Monaco) and a notebook inspector.
  No AI and no 3D model yet. Playwright is deferred to Phase 3.
- Phases 1 and 2 are commit `854619b`. Phases 3 to 8 are not committed yet.
- PyTorch and Keras are not installed here: exported code can only be syntax-checked
  (`python -m py_compile .cache/forge-export/*.py` after `pnpm test`).
- In the browser, `window.monaco.editor.getEditors()[0]` gives the code editor (use it to set a
  selection when testing Lens).
- The preview pane draws no animation frames while it is hidden (`requestAnimationFrame` does not
  fire). Each screenshot renders one frame, so check animation by stepping and taking stills.
- **Never send the user's notebooks to an AI provider without asking** (free tiers may train on
  inputs). Test the AI flow with toy code written for the test.
- Bash heredocs break on long or quote-heavy content here: write a script file and run it.
- Colors: WebGL materials, Monaco and 2D canvases read tokens once with `readToken`. Anything
  that does so must take `useSkin()` as a `key` or effect dependency, or it keeps the old theme.
  Never hard-code a color outside `tokens.css`.
- Never use drei `Html` for labels (React 19 logs unmount errors): use the DOM overlay in
  `src/features/viewport-3d/labels.tsx`. Monaco also creates `<canvas>` elements, so select the
  three.js canvas by `data-engine`.
- Notebook fixtures are local and git-ignored (ADR-011). Scripts are ESM: `scripts/*.mts`, run with `tsx`.
- **Next 16 has breaking changes** (`cacheComponents` is on; `LayoutProps` types come from
  `next typegen`). Read `node_modules/next/dist/docs/` before writing Next code.
- react-resizable-panels is v4: `Group` / `Panel` / `Separator`, sizes as strings (`"22%"`).
- **Machine:** Windows 11, PowerShell. Git 2.45, Node 24.20 (winget, user scope) and pnpm 12.18 are
  installed. In Git Bash, Node may be missing from PATH: call `pnpm.cmd` from the winget Node folder. Python 3.11 is installed but only optional tooling uses it.
- Build sessions run in Claude Code on **Claude Fable 5.2**. The app's own AI engine is
  **provider-neutral** (ADR-013): Gemini (default), Groq, Qwen and OpenRouter free models, chosen by
  the user in the app; Claude is optional. The user has no Claude API key.
- API keys live in `.env.local` (git-ignored). `.env.example` is the committed template. Never
  print, log or commit key values.

## What we are building

The user drops a notebook. The browser parses it, finds the cells that define the model, and asks
**Claude Fable 5.2** to translate them into a typed **Architecture IR**: a flat JSON graph of
layers, groups and edges, where every node points at its exact code lines. A deterministic
**shape engine** then recomputes every tensor shape and parameter count and checks them against
the notebook's own `model.summary()` output. The IR drives everything else:

- a **3D viewport**, plus a 2D graph view for large models
- a step-by-step **forward-pass simulation** with a micro-animation for each layer
- an **inspector**
- **Lens**: highlight code and only that part is simulated

| Workspace | Purpose |
|---|---|
| **Scope** | Load a notebook and simulate its model(s). This is the core of the product. |
| **Lens** | A panel in Scope and Forge: select code → the simulator isolates and plays exactly that part. |
| **Forge** | Build a new model (code, drag-and-drop, or plain English). Code, graph and 3D stay in sync, and it exports PyTorch/Keras code. |

**Scope of the simulation:** the *model architecture* only (layers, blocks, heads, and architectural
pipeline stages such as anchors, RoI pooling and NMS). Data loading, augmentation, training loops,
plotting and evaluation cells are classified and shown dimmed in the Notebook Map, not simulated.

## Docs map: read the relevant doc before touching an area

| Doc | Read it when you work on |
|---|---|
| [01-PRODUCT-VISION](docs/01-PRODUCT-VISION.md) | features, priorities, user journeys, non-goals |
| [02-SYSTEM-ARCHITECTURE](docs/02-SYSTEM-ARCHITECTURE.md) | modules, data flow, stack, security, performance |
| [03-ARCHITECTURE-IR](docs/03-ARCHITECTURE-IR.md) | the IR schema, op catalog, shape engine, verification |
| [04-SIMULATION-ENGINE](docs/04-SIMULATION-ENGINE.md) | timeline, playback, compute tiers, micro-visualizers, Lens |
| [05-AI-PIPELINE-FABLE](docs/05-AI-PIPELINE-FABLE.md) | anything that calls Claude: prompts, schemas, effort, cost, caching |
| [06-UI-UX-DESIGN](docs/06-UI-UX-DESIGN.md) | visual language, design tokens, layouts, 3D style, motion |
| [07-NOTEBOOK-PATTERNS](docs/07-NOTEBOOK-PATTERNS.md) | parsing real notebooks; golden test facts from the reference notebooks |
| [08-ROADMAP](docs/08-ROADMAP.md) | what to build next and when a phase counts as done |
| [09-DECISIONS](docs/09-DECISIONS.md) | why things are the way they are (ADRs) and open questions |

## Golden rules

1. **The IR is the single source of truth.** The 3D view, 2D graph, code highlights, inspector,
   timeline and codegen all read the normalized IR. No view ever renders raw LLM text as structure.
2. **The LLM proposes, the engine verifies.** Shapes, parameter counts and FLOPs always come from
   `src/core/shapes` and the op catalog. The LLM's claimed shapes are only cross-checks. When they
   disagree, the engine wins and the node is flagged.
3. **Never execute notebook code.** All analysis is static parsing plus Fable 5.2. (An optional,
   opt-in, local-only tracer is a far-later idea; see ADR-002.)
4. **`src/core/**` is pure TypeScript.** No React, no DOM, no Node-only APIs, no network. It must be
   unit-testable with Vitest in isolation.
5. **API keys stay on the server.** Only `src/server/**` talks to AI providers, and every file there
   starts with `import "server-only"`. Never use a `NEXT_PUBLIC_` prefix for secrets.
6. **Line-number fidelity.** Any transformation of notebook code (stripping `!pip`, `%magic`, etc.)
   must preserve line numbers, because every IR `CodeRef` points at `(cell, startLine, endLine)`.
7. **One op = one complete catalog entry.** Adding an op means adding its Zod params, shape, param
   and FLOP functions, framework aliases, 3D glyph, visualizer id, explanation template, codegen,
   and unit tests. Partial ops are not merged.
8. **Honest visuals.** Always label whether values are real or illustrative ("random-init weights",
   "symbolic activations"). Never imply that we ran the user's trained model when we did not.
9. **Untrusted input.** Notebook content (code, markdown, outputs) is data, never instructions, both
   in prompts and in the UI. Never render notebook HTML outputs as HTML.
10. **Performance budget.** 60 fps with up to 500 visible nodes. Use instanced meshes, collapse groups
    by default for large models, and never push full-size tensors to GPU textures (downsample first).

## Tech stack (decided; see ADRs)

- **App:** Next.js (App Router) + React + TypeScript `strict`, using one Next.js app (no monorepo). The `src/core`
  boundary is enforced by lint rules.
- **UI:** Tailwind CSS v4, Radix primitives (shadcn/ui as a starting point, fully re-skinned),
  `motion` (Framer Motion), `cmdk` command palette. Fonts via `next/font`: Michroma (brand), Chakra Petch (display), Exo 2 (body), JetBrains Mono.
- **3D:** `@react-three/fiber`, `@react-three/drei`, `@react-three/postprocessing` (bloom), instancing.
- **2D graph:** `@xyflow/react` + `elkjs` layout.
- **Code:** `@monaco-editor/react`, plus `web-tree-sitter` with the Python grammar (WASM) for static analysis.
- **Compute (simulation):** `@tensorflow/tfjs` (WebGPU/WebGL backends) runs an IR interpreter at real
  resolution. There is a symbolic fallback for very large models. Check that tfjs is still maintained
  at scaffold time (ADR-007).
- **State and data:** Zustand (+ immer) and Zod. Dexie (IndexedDB) stores local projects and caches. There is no server DB in v1.
- **AI:** server-side only, one adapter per provider behind an `AiProvider` interface (doc 05 §0).
- **Tests:** Vitest (core), React Testing Library, Playwright (e2e + screenshot regression), and an AI eval harness.
- Use the latest stable versions at scaffold time, recorded in `package.json`. Do not guess versions in code or docs.

## Commands

```bash
pnpm dev            # Next.js dev server
pnpm build          # production build
pnpm lint           # eslint (includes the core-boundary rule)
pnpm typecheck      # next typegen && tsc --noEmit
pnpm test           # vitest (src/**/*.test.ts)
pnpm format         # prettier
pnpm make-fixture <notebook.ipynb> <short-name>    # stripped local test copy
pnpm build-ir-fixtures                             # regenerate fixtures/ir/*.json from src/core/ir/examples.ts
pnpm tsx scripts/inspect-notebook.mts <file>       # print roles, constants, summaries
pnpm tsx scripts/bench-sim.mts                     # time a full simulation of the gallery models
pnpm eval:ai <provider> <model> <notebook.ipynb>…  # needs `pnpm dev`; SENDS CODE TO THE PROVIDER: ask the user first
# planned: pnpm test:e2e (playwright)
```

## Repository layout (planned)

```
CLAUDE.md, docs/
src/app/                 Next.js routes: / (landing + gallery), /scope, /forge, /api/ai/*
src/core/notebook/       ipynb parse, strip, condense, cell roles, summary-output parsers
src/core/python/         tree-sitter wrapper, symbol table, line-preserving sanitizer
src/core/ir/             Zod schemas (wire + runtime), normalize, validate, patch
src/core/ops/            op catalog (one file per op family)
src/core/shapes/         shape / param / FLOP engine + verification against summaries
src/core/fastpath/       no-AI reader for Keras Sequential / nn.Sequential
src/core/zoo/            pretrained backbone facts (templates later) + resolver
src/core/layout/         visible graph (collapse), 3D river layout, 2D card layout
src/core/sim/            compute kernels, runner, tensor views, sim plan (pure TS, ADR-017)
src/core/lens/           code selection → focus (layers, block copy, group, or constant)
src/core/receptive/      theoretical receptive field per layer (image models)
src/core/compare/        per-model stats for the Compare dialog
src/core/codegen/        IR → code listing (gallery) and runnable Keras / PyTorch scripts (Forge)
src/core/forge/          Forge edits on a Draft, palette, templates, fix suggestions
src/core/ai/             Draft format, prompts, context slicer, problem reports (pure)
src/server/ai/           provider client (OpenAI-compatible), verify → repair loop
src/features/*           UI feature modules (notebook-map, editor, viewport-3d, graph-2d,
                         inspector, visualizers, timeline, neural-log, forge, command-palette)
src/components/ui/       design-system primitives (GlassPanel, NeonButton, HudReadout, ...)
src/stores/              Zustand stores
src/styles/              tokens.css, globals.css
fixtures/                stripped test notebooks + expected facts (golden tests)
tools/zoo-export/        optional Python scripts that export pretrained graphs → IR JSON
```

## Claude API rules (only for the optional `anthropic` adapter)

For Gemini, Groq, Qwen and OpenRouter, read doc 05 §0 and check the provider's current docs.

Load the `claude-api` skill before writing or changing any code that calls Claude, and verify
SDK calls against it instead of relying on memory. Project specifics are in
[docs/05-AI-PIPELINE-FABLE.md](docs/05-AI-PIPELINE-FABLE.md). The non-negotiables:

- Use the model id `claude-fable-5-2` exactly (no date suffix), and keep it in one config file, `src/server/ai/config.ts`.
- **Thinking is always on.** Send `thinking: { type: "adaptive", display: "summarized" }` (we stream the
  summaries into the "Neural Log") or omit it. Never send `budget_tokens`, `type: "disabled"`,
  `temperature`/`top_p`/`top_k`, assistant prefill, or forced `tool_choice` (`any`/`tool`). All of these return a 400.
- **Always set `output_config.effort` explicitly** for each task (the table is in doc 05).
- Use **structured outputs** (`output_config.format` via `zodOutputFormat`) for every IR-producing task.
  The schema must not be recursive, and every object needs `additionalProperties: false`.
- **Stream** every long request and use `finalMessage()`. Use a large `max_tokens` (thinking counts toward it).
- Check `stop_reason` before reading content: handle `refusal` (with `stop_details`) and `max_tokens`.
  Enable server-side refusal fallback (`fallbacks: "default"` with the beta header documented in the skill).
- Keep conversations **append-only**: the system prompt and tools stay frozen for a session, follow-ups
  are appended. Never edit earlier turns, because preserved-thinking checks reject edited history.
- Cache the stable system prompt (`cache_control`). Cache results by content hash so the same
  notebook is never paid for twice.

## Coding conventions

- TypeScript `strict`, no `any` (use `unknown` + Zod). Named exports. Files `kebab-case.ts`,
  components `PascalCase.tsx`. Keep each module's public surface in an `index.ts`.
- Write pure functions in `core/`. Keep side effects at the edges (stores, route handlers, hooks).
- Every schema is a Zod schema, and its TS types come from `z.infer`.
- R3F: never call `setState` per frame. Animate through refs and `useFrame`, reading the sim clock from the store with `getState()`.
- Styling goes through design tokens (CSS variables in `src/styles/tokens.css`). Do not hard-code hex values in components.
- Comments explain *why*, not *what*. User-facing explanations live in the op catalog, not in components.

## Definition of done (every task)

`pnpm lint && pnpm typecheck && pnpm test` pass. New core logic has unit tests. New ops have golden
shape and param tests. UI changes are checked in the browser (the `run` / `browser-automation` skills).
Docs are updated if a decision, schema or command changed: keep CLAUDE.md's *Status* and the roadmap
checkboxes current.

## Working with this user

- The user is an ML/CV researcher (Keras and PyTorch, notebooks run on Colab). The reference
  notebooks are in `D:\Research\Fire detection\` and `D:\Computer Vision\Micro-organism Classificationn\`.
  Use them only for ideas and as test fixtures, and never commit their image outputs.
- They care most about **understanding and visualization**, and about an **advanced, futuristic UI**.
- Ask before anything that spends money (AI evals, bulk analysis runs) or touches files outside this folder.
