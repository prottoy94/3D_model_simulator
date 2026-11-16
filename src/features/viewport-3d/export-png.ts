import { readToken } from "@/lib/tokens";

const CAPTION_HEIGHT = 44;

/**
 * Draws the 3D view, its labels and a caption into one picture.
 * The labels are DOM elements over the WebGL canvas, so they are redrawn here as text.
 * Returns null when the stage has no WebGL canvas.
 */
export function composeSnapshot(stage: HTMLElement, caption: string): HTMLCanvasElement | null {
  // Monaco creates canvases too: three.js marks its own with `data-engine`.
  const source = stage.querySelector<HTMLCanvasElement>("canvas[data-engine]");
  if (!source || source.width === 0) return null;
  const bounds = source.getBoundingClientRect();
  const scale = source.width / bounds.width;

  const output = document.createElement("canvas");
  output.width = source.width;
  output.height = source.height + Math.round(CAPTION_HEIGHT * scale);
  const context = output.getContext("2d");
  if (!context) return null;

  context.fillStyle = readToken("--bg-deep", "#000000");
  context.fillRect(0, 0, output.width, output.height);
  context.drawImage(source, 0, 0);

  // Labels: every visible text line of the overlay, where it is on screen.
  context.textBaseline = "middle";
  context.textAlign = "center";
  for (const label of stage.querySelectorAll<HTMLElement>("[data-snapshot-label]")) {
    if (label.style.opacity === "0") continue;
    if (label.dataset.strong === "true") {
      const box = label.getBoundingClientRect();
      context.fillStyle = readToken("--glass-2", "#000000");
      context.strokeStyle = readToken("--glass-border-strong", "#888888");
      context.lineWidth = scale;
      context.beginPath();
      context.roundRect(
        (box.left - bounds.left) * scale,
        (box.top - bounds.top) * scale,
        box.width * scale,
        box.height * scale,
        6 * scale,
      );
      context.fill();
      context.stroke();
    }
    const lines = label.children.length > 0 ? [...label.children] : [label];
    for (const line of lines) {
      const text = line.textContent?.replace("▾", "").trim();
      if (!text) continue;
      const box = line.getBoundingClientRect();
      const style = getComputedStyle(line);
      // The label is scaled with a CSS transform, so its on-screen height gives the real size.
      const fontSize = Math.min(parseFloat(style.fontSize), box.height * 0.8) * scale;
      context.font = `${style.fontWeight} ${fontSize}px ${style.fontFamily}`;
      context.fillStyle = style.color;
      context.fillText(
        text,
        (box.left - bounds.left + box.width / 2) * scale,
        (box.top - bounds.top + box.height / 2) * scale,
      );
    }
  }

  // Caption: what the picture shows and where its numbers come from.
  const top = source.height;
  context.fillStyle = readToken("--bg-void", "#000000");
  context.fillRect(0, top, output.width, output.height - top);
  context.fillStyle = readToken("--glass-border-strong", "#888888");
  context.fillRect(0, top, output.width, Math.max(1, scale));
  const middle = top + (output.height - top) / 2;
  const body = getComputedStyle(document.body).fontFamily;
  context.textAlign = "left";
  context.font = `700 ${13 * scale}px ${body}`;
  context.fillStyle = readToken("--accent", "#22d3ee");
  context.fillText("NEUROSCOPE", 16 * scale, middle);
  const brandWidth = context.measureText("NEUROSCOPE").width;
  context.font = `500 ${12 * scale}px ${body}`;
  context.fillStyle = readToken("--text-2", "#cccccc");
  context.fillText(
    caption,
    16 * scale + brandWidth + 14 * scale,
    middle,
    output.width - brandWidth - 46 * scale,
  );
  return output;
}

/** Saves the snapshot as a PNG file. Everything happens in the browser. */
export function downloadSnapshot(stage: HTMLElement, caption: string, fileName: string): boolean {
  const picture = composeSnapshot(stage, caption);
  if (!picture) return false;
  picture.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
  }, "image/png");
  return true;
}
