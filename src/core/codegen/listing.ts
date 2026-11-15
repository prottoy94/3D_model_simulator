/**
 * A readable Keras or PyTorch listing of a model, one layer per line, built
 * from the op catalog's codegen. Gallery models have no notebook, so this is
 * the code their nodes link to. It is a listing for reading, not a runnable
 * script (Forge's exporter in Phase 7 produces runnable code).
 */
import type { RuntimeIR } from "../ir/runtime";
import { getOp } from "../ops";
import { shapeText } from "../shapes/tensor";

export type CodeListing = {
  text: string;
  language: "python";
  /** 1-based line of each node. Unrolled repeat copies each get their own line. */
  lineOfNode: Record<string, number>;
  /** 1-based first and last line of each group's section. */
  linesOfGroup: Record<string, [number, number]>;
};

export function generateListing(ir: RuntimeIR, modelId: string): CodeListing {
  const model = ir.models.find((candidate) => candidate.id === modelId);
  const framework = ir.engineFramework;
  const lines: string[] = [];
  const lineOfNode: Record<string, number> = {};
  const linesOfGroup: Record<string, [number, number]> = {};

  lines.push(
    `# ${model?.name ?? "Model"} · generated ${framework === "keras" ? "Keras" : "PyTorch"} listing`,
  );
  lines.push("# One line per layer, in the order data flows. Shapes are computed by the engine.");

  let currentGroup: string | null = null;
  for (const nodeId of ir.index.topoOrder[modelId] ?? []) {
    const node = ir.nodes[nodeId]!;
    const group = ir.groups[node.groupId];
    if (group && group.id !== currentGroup) {
      currentGroup = group.id;
      if (group.kind !== "model") {
        lines.push("");
        lines.push(`# ── ${group.label} ──`);
      } else lines.push("");
    }

    const spec = getOp(node.op);
    let call = `# ${node.op} (not in the catalog yet)`;
    if (spec && node.outShape && node.status !== "error") {
      try {
        const io = { in: node.inShapes, out: node.outShape };
        call =
          framework === "keras"
            ? spec.codegen.keras(node.params, io)
            : spec.codegen.pytorch(node.params, io);
      } catch {
        call = `# ${spec.title}`;
      }
    }
    const shape = node.outShape ? `  # → ${shapeText(node.outShape)}` : "";
    lines.push(`${call}${shape}`);
    lineOfNode[nodeId] = lines.length;

    // Extend the section of every group this node sits in.
    for (let id: string | null = node.groupId; id && ir.groups[id]; id = ir.groups[id]!.parentId) {
      const range = linesOfGroup[id];
      linesOfGroup[id] = range ? [range[0], lines.length] : [lines.length, lines.length];
    }
  }
  return { text: lines.join("\n"), language: "python", lineOfNode, linesOfGroup };
}
