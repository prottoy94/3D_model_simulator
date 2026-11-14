import type { PythonParser } from "@/core/notebook";

let cached: Promise<PythonParser> | null = null;

/**
 * Loads the tree-sitter Python parser in the browser (once). The two WASM files
 * are served from /public/wasm; they are copies of the files shipped in the
 * `web-tree-sitter` and `tree-sitter-python` packages.
 */
export function getPythonParser(): Promise<PythonParser> {
  cached ??= (async () => {
    const { Language, Parser } = await import("web-tree-sitter");
    await Parser.init({ locateFile: (file: string) => `/wasm/${file}` });
    const language = await Language.load("/wasm/tree-sitter-python.wasm");
    const parser = new Parser();
    parser.setLanguage(language);
    return parser;
  })();
  // A failed load (e.g. offline) should be retried on the next attempt.
  cached.catch(() => {
    cached = null;
  });
  return cached;
}
