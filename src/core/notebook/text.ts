/** Text clean-up helpers for notebook outputs. Pure string functions, no DOM. */

// CSI sequences (colors, cursor moves) and OSC sequences (terminal hyperlinks).
const ANSI_PATTERN = /\u001b\[[0-9;?]*[ -/]*[@-~]|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)/g;

export function stripAnsi(text: string): string {
  return text.replace(ANSI_PATTERN, "");
}

const ENTITIES: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: " ",
};

function decodeEntities(text: string): string {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, body: string) => {
    if (body[0] === "#") {
      const hex = body[1] === "x" || body[1] === "X";
      const code = hex ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
      return Number.isFinite(code) ? String.fromCodePoint(code) : match;
    }
    return ENTITIES[body.toLowerCase()] ?? match;
  });
}

/**
 * Converts an HTML fragment (typically a pandas table) into plain text.
 * Table rows become lines with " | " between cells. The HTML is never rendered.
 */
export function htmlToText(html: string): string {
  const text = html
    .replace(/<(script|style)\b[\s\S]*?<\/\1>/gi, "")
    .replace(/<\/(td|th)>\s*<(td|th)\b[^>]*>/gi, " | ")
    .replace(/<\/(tr|p|div|li|h[1-6]|pre|table|thead|tbody)>/gi, "\n")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<[^>]+>/g, "");
  return decodeEntities(text)
    .split("\n")
    .map((line) => line.replace(/[ \t]+/g, " ").trim())
    .filter((line) => line.length > 0)
    .join("\n");
}

/**
 * Keeps the head and tail of a very long text. Training logs are long, but the
 * first and last lines are what matter.
 */
export function truncateMiddle(
  text: string,
  maxChars: number,
): { text: string; truncated: boolean } {
  if (text.length <= maxChars) return { text, truncated: false };
  const head = Math.floor(maxChars * 0.6);
  const tail = maxChars - head;
  const omitted = text.length - maxChars;
  return {
    text: `${text.slice(0, head)}\n… [${omitted} characters omitted] …\n${text.slice(-tail)}`,
    truncated: true,
  };
}
