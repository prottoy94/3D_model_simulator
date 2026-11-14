import type { CellRole } from "@/core/notebook";

/** How each cell role is named and coloured across the UI. */
export const ROLE_META: Record<CellRole, { label: string; color: string; description: string }> = {
  model: {
    label: "model",
    color: "var(--role-model)",
    description: "Defines architecture. This is what gets simulated.",
  },
  compile: {
    label: "setup",
    color: "var(--role-compile)",
    description: "Training setup: compile, optimizer, callbacks, freezing layers.",
  },
  train: { label: "train", color: "var(--role-train)", description: "Runs training." },
  eval: { label: "eval", color: "var(--role-eval)", description: "Evaluates or predicts." },
  data: { label: "data", color: "var(--role-data)", description: "Loads or prepares data." },
  viz: { label: "plot", color: "var(--role-viz)", description: "Draws figures." },
  config: { label: "config", color: "var(--role-config)", description: "Constants and settings." },
  imports: { label: "imports", color: "var(--role-imports)", description: "Imports libraries." },
  setup: { label: "shell", color: "var(--role-setup)", description: "Shell or magic commands." },
  other: { label: "code", color: "var(--role-other)", description: "Other code." },
  markdown: { label: "text", color: "var(--role-markdown)", description: "Markdown text." },
};
