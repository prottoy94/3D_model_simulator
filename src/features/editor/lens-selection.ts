import { resolveSelection, type CodeRange, type LensFocus, type LensInterval } from "@/core/lens";
import { useModelStore } from "@/stores/model-store";
import { useProjectStore } from "@/stores/project-store";

/**
 * What a selection of editor lines would focus. Editor lines are converted to
 * (cell, line) for a notebook; a gallery model's generated listing has one
 * line per layer, so its lines are matched directly.
 */
export function focusForEditorLines(startLine: number, endLine: number): LensFocus | null {
  const { ir, activeModelId, listing } = useModelStore.getState();
  if (!ir || !activeModelId) return null;

  if (listing) {
    const intervals: LensInterval[] = [];
    for (const [nodeId, line] of Object.entries(listing.lineOfNode)) {
      intervals.push({ cell: -1, startLine: line, endLine: line, role: "declare", nodeId });
    }
    for (const [groupId, [first, last]] of Object.entries(listing.linesOfGroup)) {
      intervals.push({ cell: -1, startLine: first, endLine: last, role: "definition", groupId });
    }
    return resolveSelection(ir, activeModelId, {
      ranges: [{ cell: -1, startLine, endLine }],
      intervals,
    });
  }

  const project = useProjectStore.getState().project;
  if (!project) return null;
  const ranges: CodeRange[] = [];
  for (const cell of project.document.cells) {
    if (cell.kind !== "code" || cell.lastLine < startLine || cell.firstLine > endLine) continue;
    ranges.push({
      cell: cell.index,
      startLine: Math.max(startLine, cell.firstLine) - cell.firstLine + 1,
      endLine: Math.min(endLine, cell.lastLine) - cell.firstLine + 1,
    });
  }
  if (ranges.length === 0) return null;
  // Constants assigned inside the selection, e.g. `PATCH_SIZE = 16`.
  const symbols = project.analysis.symbols
    .filter(
      (symbol) =>
        symbol.kind === "assignment" &&
        ranges.some(
          (range) =>
            range.cell === symbol.cell &&
            range.startLine <= symbol.endLine &&
            symbol.startLine <= range.endLine,
        ),
    )
    .map((symbol) => symbol.name);
  return resolveSelection(ir, activeModelId, { ranges, symbols });
}
