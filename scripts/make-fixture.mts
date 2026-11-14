/**
 * Creates a small, text-only copy of a notebook for tests.
 *
 *   pnpm make-fixture <path/to/notebook.ipynb> <short-name>
 *
 * Writes fixtures/notebooks/<short-name>.ipynb with code, markdown and text
 * outputs only. Images and widgets are removed. These copies are git-ignored
 * by default because notebooks can be private (see fixtures/README.md).
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { parseNotebook, toStrippedIpynb } from "../src/core/notebook/parse";

const [source, name] = process.argv.slice(2);
if (!source || !name) {
  console.error("usage: pnpm make-fixture <notebook.ipynb> <short-name>");
  process.exit(1);
}

const raw = readFileSync(source, "utf8");
const notebook = parseNotebook(raw);
const stripped = toStrippedIpynb(notebook);

const outDir = path.join(process.cwd(), "fixtures", "notebooks");
mkdirSync(outDir, { recursive: true });
const outFile = path.join(outDir, `${name}.ipynb`);
writeFileSync(outFile, stripped, "utf8");

const kb = (bytes: number) => `${(bytes / 1024).toFixed(0)} KB`;
console.log(
  `${name}: ${kb(raw.length)} -> ${kb(stripped.length)} | ` +
    `${notebook.stats.codeCells} code cells, ${notebook.stats.markdownCells} markdown cells, ` +
    `${notebook.stats.droppedImages} images removed`,
);
