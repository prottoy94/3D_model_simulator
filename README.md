# NeuroScope

**See how your neural network works, in 3D.**

NeuroScope is a website. You give it a Jupyter notebook (`.ipynb`) that contains a machine
learning model. It reads the model, draws it as a 3D scene, and lets you watch data move through
it layer by layer.

It is made for people who write models and want to _understand_ them, not only run them.

---

## What you can do with it

| You want to…                          | NeuroScope does this                                                    |
| ------------------------------------- | ----------------------------------------------------------------------- |
| See the shape of your model           | Draws every layer as a 3D block, colored by its type                    |
| Know what each layer does             | Click a block to read a plain explanation, its sizes and its parameters |
| Watch the model work                  | Press play: a sample goes through the model, one layer at a time        |
| Look at one part only                 | Select some code and press **Lens**: only that part is shown and played |
| Build a new model                     | Use **Forge**: add layers by clicking, then export Keras or PyTorch     |
| Compare two models                    | Press **Compare** to see them side by side                              |

---

## How it works

Think of it as five steps.

```
 Your notebook  →  Read  →  Understand  →  Check  →  Show and simulate
```

**1. Read.** You drop a notebook on the page. It is opened in your browser. NeuroScope looks at
each cell and decides what it is: model code, training code, data loading, plots and so on. Only
the model code matters for the next steps.

**2. Understand.** The model code is turned into a simple list: _these are the layers, and this
is how they connect_. Simple models (a plain stack of layers) are read directly. For more complex
code, an AI model reads it and writes that list.

**3. Check.** NeuroScope never trusts the AI's numbers. Its own engine works out the output size
and the parameter count of every layer. If your notebook has a saved `model.summary()`, the
engine compares its numbers with it and tells you whether they match.

**4. Show.** The checked list is drawn as a 3D scene and as a flat 2D graph. Every block is
linked to the line of code that created it. Click a block to jump to the code, or click the code
to light up the block.

**5. Simulate.** Press play. A sample picture is passed through the model and you see the real
result of each layer.

### One important note about the simulation

Your notebook's code is **never run**, and your trained weights are **not used**. The simulation
uses random starting weights. So it shows _how_ data flows and changes shape, not what your
trained model would predict. The app always says this on screen.

---

## The three workspaces

- **Scope**: open a notebook, look at its model, simulate it. This is the main screen.
- **Lens**: a tool inside Scope. Select code to focus on just that part of the model.
- **Forge**: build your own model from layers, see it in 3D right away, and export the code.

There is also a **gallery** on the home page with ready-made models (AlexNet, VGG16, a small
Vision Transformer, FireNet and a small object detector), so you can try the app without a
notebook.

---

## Run it on your computer

You need [Node.js](https://nodejs.org) (version 20 or newer) and
[pnpm](https://pnpm.io/installation).

```bash
pnpm install
```

```bash
pnpm dev
```

Then open <http://localhost:3000> in your browser.

### Turn on the AI step (optional)

The gallery, simple notebooks and Forge work without AI. To read complex notebooks you need a
free API key from one of these: Google Gemini, Groq, OpenRouter or Qwen.

1. Copy the file `.env.example` and name the copy `.env.local`.
2. Paste your key into `.env.local`.
3. Restart `pnpm dev`.

`.env.local` is ignored by git, so your keys are never uploaded to GitHub. The keys stay on the
server and are never sent to the browser.

---

## Your privacy

- Your notebook is opened **in your browser**. Images and outputs are removed there.
- Nothing is sent to an AI provider unless you press **Analyze with AI** and agree.
- When you agree, only the model code is sent, not your data or results.
- There is a **Local only** switch that blocks every AI request.

---

## What it is built with

| Part              | Tool                                   |
| ----------------- | -------------------------------------- |
| Website           | Next.js, React, TypeScript             |
| 3D scene          | three.js (React Three Fiber)           |
| 2D graph          | React Flow                             |
| Code viewer       | Monaco (the editor from VS Code)       |
| Styling           | Tailwind CSS                           |
| Simulation        | Our own math code, running in a worker |
| Tests             | Vitest                                 |

---

## Project folders

```
src/app/         the pages (home, Scope, Forge) and the AI server routes
src/core/        the logic: reading notebooks, layers, sizes, simulation
src/features/    the screens: 3D view, 2D graph, code editor, inspector, Forge
src/stores/      the app's state
src/server/      the only code that talks to AI providers
docs/            the design documents and the roadmap
```

Everything in `src/core` is plain logic with no screen code, so it can be tested on its own.

---

## Useful commands

```bash
pnpm dev         # start the website
pnpm test        # run the tests
pnpm lint        # check the code style
pnpm typecheck   # check the types
pnpm build       # make a production build
```

---

## What works today, and what does not

**Works**

- Reading notebooks and sorting their cells
- 50 layer types: convolution, pooling, dense, attention, normalization, recurrent layers
  (RNN, LSTM, GRU), and object-detection steps (anchors, box decoding, NMS, RoI pooling)
- Checking sizes and parameter counts against the notebook's saved summary
- 3D view, 2D graph, and links between code and model
- Forward-pass simulation with random starting weights
- Lens, Forge, Compare, a light theme, colorblind-safe colors and picture export

**Not built yet**

- Using your real trained weights
- Classical machine learning models (decision trees, k-means, SVM)
- Looking inside a pretrained backbone layer by layer
- Replaying a training run, and video export

The full plan is in [docs/08-ROADMAP.md](docs/08-ROADMAP.md).

---

## Learn more

The [docs](docs) folder explains every part in detail: the product idea, the system design, the
model format, the simulation, the AI step and the look of the interface.
