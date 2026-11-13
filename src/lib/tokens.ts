/**
 * Reads a design token from CSS at runtime. WebGL materials and 2D canvases
 * cannot use `var(--x)`, and components must not hard-code hex values.
 */
export function readToken(name: `--${string}`, fallback = "#ffffff"): string {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}
