/**
 * Makes notebook code parseable as plain Python WITHOUT changing line numbers.
 *
 * Notebooks contain IPython-only syntax (`!pip install x`, `%matplotlib inline`,
 * `%%time`, `obj?`). Each such line is replaced in place by a valid statement
 * that keeps the original text as a comment. Every CodeRef points at
 * (cell, line), so lines are never added or removed (golden rule 6).
 */
export function sanitizePython(source: string): string {
  const lines = source.split("\n");
  let openTripleQuote: string | null = null;

  return lines
    .map((line) => {
      // Leave anything inside a multi-line string alone: "!" there is just text.
      const wasInsideString = openTripleQuote !== null;
      openTripleQuote = trackTripleQuotes(line, openTripleQuote);
      if (wasInsideString) return line;

      const indent = line.match(/^\s*/)?.[0] ?? "";
      const body = line.slice(indent.length);

      // Shell commands, line magics, cell magics, and help requests ("?name", "name?").
      if (/^[!%]/.test(body) || /^\?/.test(body) || /^[\w.]+\?\??\s*$/.test(body)) {
        return `${indent}pass  # ${body}`;
      }
      // Captured shell output or magic: `files = !ls`, `t = %timeit f()`.
      const assigned = body.match(/^([\w.,\s[\]]+?)=\s*([!%].*)$/);
      if (assigned) {
        return `${indent}${assigned[1].trimEnd()} = None  # ${assigned[2]}`;
      }
      return line;
    })
    .join("\n");
}

/** Returns the triple-quote delimiter still open after this line, or null. */
function trackTripleQuotes(line: string, open: string | null): string | null {
  let state = open;
  let i = 0;
  while (i < line.length) {
    if (state) {
      const close = line.indexOf(state, i);
      if (close < 0) return state;
      i = close + 3;
      state = null;
      continue;
    }
    const char = line[i];
    if (char === "#") return null; // rest of the line is a comment
    const triple = line.slice(i, i + 3);
    if (triple === '"""' || triple === "'''") {
      state = triple;
      i += 3;
      continue;
    }
    if (char === '"' || char === "'") {
      // Skip a single-line string so quotes inside it are not misread.
      let j = i + 1;
      while (j < line.length && line[j] !== char) j += line[j] === "\\" ? 2 : 1;
      i = j + 1;
      continue;
    }
    i += 1;
  }
  return state;
}
