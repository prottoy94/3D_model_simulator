# 06 — UI / UX Design: "Holographic Lab"

## 1. Design direction

**Mood:** a holographic research lab at night. Deep-space darkness, glass instrument panels
floating over a live 3D specimen (the model), and neon light that *means something*: every color
encodes an operation family, and light moves only when data moves.

**Principles**
1. **Light = data.** Glow, pulses and bloom happen only when a tensor flows. Idle UI stays calm
   and dim, so motion always carries meaning.
2. **The specimen is the hero.** The 3D viewport fills most of the screen. Panels are translucent
   glass that can be collapsed and never fully hides the scene.
3. **Every number is clickable.** Any shape, param count or symbol links to its code and its explanation.
4. **Futuristic but legible.** Body text stays high-contrast and calm. Effects decorate the frame, never the text.
5. **Honest instruments.** Readouts show their source (✓ summary, ≈ computed, ⚠ mismatch, "illustrative").

## 2. Design tokens (`src/styles/tokens.css`)

### Color (dark is the primary theme)
```css
:root {
  /* space */
  --bg-void:        #000000;   /* page background */
  --bg-deep:        #000000;   /* viewport base */
  --grid-line:      rgba(90, 140, 255, 0.06);
  /* glass */
  --glass-1:        rgba(1, 2, 5, 0.8);      /* panels (with backdrop-filter: blur(18px) saturate(140%)) */
  --glass-2:        rgba(4, 6, 12, 0.9);      /* raised / popovers */
  --glass-border:   rgba(130, 190, 255, 0.14);
  --glass-highlight: linear-gradient(135deg, rgba(120,220,255,.35), rgba(232,121,249,.12) 40%, transparent 70%);
  /* text */
  --text-1:         #E8F1FF;   /* primary */
  --text-2:         #9DB0CC;   /* secondary */
  --text-3:         #5E6F8C;   /* muted / disabled */
  /* brand + state */
  --accent:         #22D3EE;   /* primary action, focus ring, data flow */
  --accent-2:       #E879F9;   /* secondary highlight */
  --ok:             #A3E635;
  --warn:           #FBBF24;
  --error:          #FB7185;
  --frozen:         #93C5FD;
  /* operation families (shared by 3D, 2D, chips, timeline) */
  --op-io:      #CBD5E1;  --op-conv:   #22D3EE;  --op-pool:   #A78BFA;  --op-norm:    #2DD4BF;
  --op-act:     #A3E635;  --op-linear: #FBBF24;  --op-attn:   #E879F9;  --op-recur:   #38BDF8;
  --op-shape:   #94A3B8;  --op-merge:  #F8FAFC;  --op-reg:    #64748B;  --op-det:     #FB923C;
  --op-classic: #FACC15;  --op-custom: #FB7185;
}
```
- A **light theme** ("Lab Daylight": white glass, deeper saturated family colors) arrives later.
  All colors are tokens from day one so it is a pure token swap. Theme switching follows
  `prefers-color-scheme` with a manual override (`data-theme`).
- A **colorblind-safe** family palette is available as an alternative token set. Shape and glyph
  differ per family too, so color is never the only cue.
- Contrast: `--text-1` and `--text-2` on `--glass-1` over `--bg-void` must meet WCAG AA. Check
  with an automated contrast test on the token pairs.

### Typography
| Role | Font | Use |
|---|---|---|
| Brand | **Michroma** 400, wide tracking | The wordmark only |
| Display | **Chakra Petch** 500–700, uppercase with wide tracking for HUD labels | Headings, panel titles, mode switch |
| UI / body | **Exo 2** 400–600 | Everything readable |
| Mono | **JetBrains Mono** 400–500, tabular numbers | Code, shapes (`54×54×96`), params, telemetry |

Scale: 11 / 12 / 13 / 14 / 16 / 20 / 28 / 40 px. The UI is dense, so the default body size is 13–14 px.

### Space, radius, elevation
- 4 px base grid. Panel padding 12–16 px. Panel radius 14 px, controls 8 px, chips 999 px.
- Elevation comes from glass opacity, the border highlight and a soft outer glow
  (`0 0 0 1px var(--glass-border), 0 8px 40px rgba(0,0,0,.45)`), never heavy drop shadows.

### Motion
- Springs (motion): panels `stiffness 260, damping 28`, and the camera uses critically damped springs.
- Durations: micro 120 ms, UI 220 ms, layout and camera 600–900 ms.
- **Reduced motion:** with `prefers-reduced-motion` (or the in-app toggle), particles are
  replaced by static highlighted edges, camera moves become cuts, and bloom intensity drops.
  The simulation still steps, with crossfades.

## 3. App shell and layouts

### 3.1 Landing (`/`)
- A full-bleed animated **neural constellation** (instanced points and lines, drifting slowly; it
  reacts to the cursor) behind a centered **drop zone**: "Drop a notebook to see its mind." A
  dashed holographic frame "charges up" on drag-over.
- Below it, the **Model Gallery**: glass cards (AlexNet, VGG16, ResNet50, MobileNetV2,
  EfficientNetB0, ViT, LSTM, Transformer, U-Net, YOLO). Each card has a mini rotating 3D
  thumbnail. A click opens Scope with the prebuilt IR (no API call).
- Footer chips: "Local-only mode", "Privacy", "Keyboard shortcuts".

### 3.2 Scope workspace (`/scope`)
```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ ◈ NEUROSCOPE  │ SCOPE · FORGE │ alexnet_vgg_scratch.ipynb ▾ │ Model: AlexNet ▾ │ ⌘K │ ◉ Fable 5.2 ▾ │
├──────────────────┬───────────────────────────────────────────────────────┬────────────────┤
│ NOTEBOOK MAP     │                                                       │ INSPECTOR      │
│ ▏c2  imports     │                                                       │ CONV2D · n_conv1│
│ ▏c4  config      │                 3D VIEWPORT                           │ 96 @ 11×11 / 4 │
│ ▶c14 MODEL ◀     │   ▣ in ─▶ ▣ conv1 ─▶ ▢ bn ─▶ ▤ pool ─▶ ▣ conv2 ─▶ …   │ in  224×224×3  │
│ ▏c16 compile     │                                                       │ out  54×54×96 ✓│
│ ▏c20 train  ░    │                                                       │ params 34,944 ✓│
│ ▏c24 eval   ░    │                                                       │ FLOPs 203.2 M  │
│──────────────────│                                                       │ ┌────────────┐ │
│ CODE  cell 14    │                                                       │ │ micro-viz  │ │
│  11 # --- Conv 1 │                                                       │ │ (kernel    │ │
│▌12 Conv2D(96,(11 │                                                       │ │  sweep)    │ │
│  13 BatchNorm…   │  [3D|2D]  [Expand all]  [Labels]  [Camera: Follow ▾]  │ └────────────┘ │
│                  │  ◌ Real compute · random-init weights                 │ ✦ Ask Fable    │
├──────────────────┴───────────────────────────────────────────────────────┴────────────────┤
│ ⏮ ◀ ▶ ▶▶ ⏭ │ ●──●──●──●───────●────●────●──────●──── │ step 2/14 · conv1 · phase: sweep │ 1.0× │ ⟲ │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```
- **Left rail (resizable, collapsible):** the Notebook Map on top (cells as slim bars, role color
  on the left edge, model cells glowing, non-architecture cells dimmed `░`) and the Monaco editor
  below (the selected cell, or all cells as a continuous document with cell separators).
- **Center:** the viewport. Floating toolbar at the bottom-left of the viewport. The compute
  tier label is always visible.
- **Right rail:** the Inspector. Tabs: *Layer* (default), *Tensor* (channel grid, histogram,
  stats), *Code* (refs), *Explain* (AI tutor thread).
- **Bottom:** the Timeline. Step markers are colored by op family, macro-steps are wider, and
  group bands sit above the markers (Block 1, Block 2, ...). Scrubbing previews the step label.
- **Neural Log drawer** (slides up from the bottom, or toggle with `~`): a terminal-style stream
  of Fable's reasoning summaries and pipeline stages
  `PARSE ✓ · LOCATE ✓ · EXTRACT ⟳ · VERIFY · REPAIR · LAYOUT`, with token and cost readouts.

### 3.3 Lens (inside Scope and Forge)
- Selecting code shows a floating chip near the selection: **◎ Simulate selection** (and
  **✦ Explain**). Shortcut `F`.
- In focus mode, the viewport dims outside the subgraph, a holographic focus frame appears with
  the label "LENS · transformer_block · canonical instance (×4)", and the timeline is replaced by
  the focused mini-timeline. `Esc` exits.
- Unmapped selection: the chip changes to **✦ Analyze selection with Fable · ≈ $0.04**.

### 3.4 Forge workspace (`/forge`)
```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ ◈ NEUROSCOPE │ SCOPE · FORGE │ untitled-model ▾ │ ✦ "Describe a model…"  [Generate] │ ⌘K │
├───────────────┬──────────────────────────────────────────────┬───────────────────────────┤
│ PALETTE       │   GRAPH (2D editable)  ⇄  3D PREVIEW (split)  │ CODE  [PyTorch|Keras]     │
│ ▸ Conv        │   ┌──────┐   ┌──────┐   ┌──────┐               │ class Net(nn.Module):     │
│ ▸ Pool        │   │Input │──▶│Conv2d│──▶│ReLU  │──▶ …          │   def __init__(self):     │
│ ▸ Norm        │   └──────┘   └──────┘   └──────┘               │     self.c1 = nn.Conv2d(… │
│ ▸ Attention   │       ⚠ shape mismatch: Linear expects 6400,    │                           │
│ ▸ Templates   │         got 9216 (fix: Flatten / change in)     │                           │
├───────────────┴──────────────────────────────────────────────┴───────────────────────────┤
│ Σ params 2.31 M · FLOPs 0.71 G · memory 9.2 MB · depth 12 · receptive field 67 px │ ▶ Simulate │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```
- Drag layers from the palette. Edge connections validate live. A shape error shows a rose
  pulsing edge with a fix suggestion. Code and graph are bidirectionally synced (code edits are
  parsed by the fast path, or by Fable for complex code).
- Templates: ResNet block, MBConv, Transformer encoder, U-Net level, RPN head, CBAM.

### 3.5 Command palette (`⌘K` / `Ctrl+K`)
Jump to node, layer type or cell; switch model; toggle 2D/3D; play from here; expand a group;
change camera; open a gallery model; export.

## 4. 3D visual language

| Element | Look |
|---|---|
| **Feature-map tensor** (conv, pool, norm, act) | A frosted-glass box. Height and depth follow log(H), log(W); width follows log(C). A family-colored inner glow. During play, sampled channel thumbnails sit stacked inside like film slides. |
| **Vector** (dense, GAP output) | A column of instanced neuron spheres (capped at 64 visible + "… 4096" ellipsis), with a faint wire link to the previous layer during play. |
| **Token sequence** (ViT, NLP) | A row of thin token tiles (196 patches → a 14×14 tile carpet that unfolds into a line), with embedding height bars. |
| **Attention** | A magenta ring of tokens with arcs whose opacity is the attention weight. The head selector rotates between rings. |
| **Merge** | A small white octahedron "junction" where streams meet. |
| **Reshape/flatten** | A short animated "ribbon" connector. |
| **Recurrent** | A cell glyph with a looping arrow; it unrolls into a time chain when expanded. |
| **Detection stages** | Orange wireframe glyphs: a grid of anchors, a box-in-box for RoI, a funnel for NMS. |
| **Group** | A holographic bounding frame with corner brackets and a label tab. Collapsed groups are a solid capsule labelled `Stage 4 · 3× MBConv · 112→…`. Repeat groups show stacked ghost copies with `×N`. |
| **Frozen** | Desaturated with a frost shader, a lock glyph and ice-blue edges. Thawing animates (phase 2). |
| **Edges** | Thin curved tubes in the producer's color at 25% opacity. While data flows, bright particles travel along them. Residual and skip connections arc above the main river. |
| **Status** | ⚠ mismatch: a rose outline plus a pulsing badge. `ai-synth`: a dashed outline. Low confidence: an amber dotted outline. |

Scene: an infinite faint grid floor, an exponential fog that fades to `--bg-deep`, one key light
plus environment lighting, bloom only on emissive materials, and an optional subtle chromatic
aberration during camera moves (off when reduced motion is on).

## 5. Key components (`src/components/ui`, `src/features/*`)

`GlassPanel`, `PanelHeader` (HUD label + actions), `NeonButton` (primary/ghost/danger), `Chip`
(op family, status, provenance), `HudReadout` (label + mono value + source badge), `ShapeTag`
(`54×54×96`, color-coded dims), `Kbd`, `SegmentedControl` (SCOPE/FORGE, 3D/2D), `Timeline`,
`StepMarker`, `CodeRefChip`, `CostEstimate`, `NeuralLog`, `AssumptionsPanel` (editable symbols
with provenance), `ModelPicker`, `CompareSplit` (P2).

## 6. States

| State | Treatment |
|---|---|
| Empty (no notebook) | The landing drop zone and gallery. |
| Parsing (< 1 s) | The cell map fills top to bottom with a scanline. |
| Heuristic preview | A grey "draft" 3D skeleton from heuristics, labelled "Draft · awaiting Fable". |
| Analyzing | The Neural Log auto-opens, cells being read shimmer, and the draft skeleton gets a progress halo. Cancel is available. |
| Verified | The skeleton "crystallizes" into family colors (a 600 ms wave from input to output). The ✓ count shows in the top bar. |
| Partially verified | ⚠ badges, and the Assumptions panel opens with the affected symbols highlighted. |
| Error / refusal | An inline glass alert with the cause, a retry button and Local-only mode. Never a blank screen. |
| No WebGL/WebGPU | The 2D graph plus a static inspector with a symbolic-compute notice. |

## 7. Accessibility

- Full keyboard navigation. The viewport is focusable, arrow keys move between nodes in
  topological order, Enter opens the inspector, and Space plays or pauses.
- The 2D graph view is the accessible equivalent of the 3D scene, and every node has an ARIA
  label ("Conv2D, 96 filters, 11 by 11, stride 4, output 54 by 54 by 96").
- Visible focus rings (`--accent`, 2 px plus glow). Hit targets are at least 32 px.
- Captions: step narration is always available as text in the inspector, and also as a
  live-region announcement while stepping.
- Reduced motion and colorblind palettes (above).

## 8. Responsiveness

Desktop-first: the full workspace needs ≥ 1280 px. From 1024 to 1279 px the rails become
overlay drawers. Below 1024 px there is a read-only "viewer" with the gallery, play, inspector as a
bottom sheet, and no editing. No horizontal page scroll at any width.

## 9. Sound (optional, off by default)
Soft UI ticks for steps and a low "hum" when data flows, toggled in settings. They add presence
in demos and are never required.
