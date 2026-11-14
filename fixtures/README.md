# Fixtures

| Folder | Committed? | What |
|---|---|---|
| `summaries/` | Yes | Raw `model.summary()` text captured from real notebooks. Parser test inputs. They contain layer tables only, no code or data. |
| `notebooks/` | **No (git-ignored)** | Stripped, text-only copies of the reference notebooks, made with `pnpm make-fixture`. They contain the notebooks' full source code, which may be private or unpublished, so they stay on this machine. |

Tests that need `notebooks/` skip themselves when a fixture is missing, so the suite still passes
on a fresh clone.

To recreate a notebook fixture:

```bash
pnpm make-fixture "D:\path\to\notebook.ipynb" short-name
```

The short names the tests expect are listed in `src/core/notebook/reference.test.ts`.

To commit a notebook fixture on purpose (for example a public tutorial notebook), add an
exception for that file in `.gitignore`.
