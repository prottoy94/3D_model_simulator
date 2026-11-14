/** Dev helper: saves the joined text outputs of every cell that prints a model summary. */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { parseNotebook } from "../src/core/notebook/parse";

const dir = path.join(process.cwd(), "fixtures", "notebooks");
const out = path.join(process.cwd(), "fixtures", "summaries");
mkdirSync(out, { recursive: true });
for (const file of readdirSync(dir).filter((f) => f.endsWith(".ipynb"))) {
  const nb = parseNotebook(readFileSync(path.join(dir, file), "utf8"));
  for (const cell of nb.cells) {
    const text = cell.outputs.map((o) => o.text).join("\n");
    if (!/Total params|Param #/.test(text)) continue;
    const name = `${file.replace(".ipynb", "")}-cell${cell.index}.txt`;
    writeFileSync(path.join(out, name), text, "utf8");
    console.log(name, text.length);
  }
}
