/**
 * Loads the tree-sitter Python parser in Node, for tests and scripts.
 * The browser loads the same WASM files from /public/wasm (src/lib/python-parser.ts).
 */
import path from "node:path";
import { Language, Parser } from "web-tree-sitter";

let cached: Promise<Parser> | null = null;

export function loadPythonParserForNode(): Promise<Parser> {
  cached ??= (async () => {
    await Parser.init();
    const language = await Language.load(
      path.join(process.cwd(), "public", "wasm", "tree-sitter-python.wasm"),
    );
    const parser = new Parser();
    parser.setLanguage(language);
    return parser;
  })();
  return cached;
}
