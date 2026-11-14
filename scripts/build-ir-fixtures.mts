/**
 * Writes fixtures/ir/*.json from the hand-written examples in
 * src/core/ir/examples.ts. Run with `pnpm build-ir-fixtures` after editing them.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { EXAMPLE_IRS } from "../src/core/ir/examples";

const dir = path.join(process.cwd(), "fixtures", "ir");
mkdirSync(dir, { recursive: true });
for (const [name, build] of Object.entries(EXAMPLE_IRS)) {
  const file = path.join(dir, `${name}.json`);
  writeFileSync(file, JSON.stringify(build(), null, 2) + "\n");
  console.log(`wrote ${path.relative(process.cwd(), file)}`);
}
