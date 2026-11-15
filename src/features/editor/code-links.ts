import type { CodeListing } from "@/core/codegen/listing";
import type { RuntimeIR } from "@/core/ir";
import { editorLineFor, type NotebookDocument } from "@/core/notebook/document";

/** A range of editor lines that belongs to a node or a group of the model on screen. */
export type CodeLink = { id: string; kind: "node" | "group"; start: number; end: number };

/**
 * Maps the active model onto editor lines: through CodeRefs for a notebook,
 * or through the generated listing for a gallery model.
 */
export function buildCodeLinks(
  ir: RuntimeIR | null,
  modelId: string | null,
  document: NotebookDocument | null,
  listing: CodeListing | null,
): CodeLink[] {
  if (!ir || !modelId) return [];
  const links: CodeLink[] = [];
  const nodeIds = ir.index.topoOrder[modelId] ?? [];
  const groups = Object.values(ir.groups).filter((group) => group.modelId === modelId);

  if (listing) {
    for (const id of nodeIds) {
      const line = listing.lineOfNode[id];
      if (line) links.push({ id, kind: "node", start: line, end: line });
    }
    for (const group of groups) {
      const range = listing.linesOfGroup[group.id];
      if (range && group.kind !== "model")
        links.push({ id: group.id, kind: "group", start: range[0], end: range[1] });
    }
    return links;
  }
  if (!document) return [];

  const add = (id: string, kind: CodeLink["kind"], refs: RuntimeIR["nodes"][string]["refs"]) => {
    for (const ref of refs) {
      const start = editorLineFor(document, ref.cell, ref.startLine);
      const end = editorLineFor(document, ref.cell, ref.endLine);
      if (start !== null && end !== null)
        links.push({ id, kind, start, end: Math.max(start, end) });
    }
  };
  for (const id of nodeIds) add(id, "node", ir.nodes[id]!.refs);
  for (const group of groups) if (group.kind !== "model") add(group.id, "group", group.refs);
  return links;
}

/**
 * What a line of code stands for: the ids with the tightest range around it.
 * Nodes win over groups, and one line can map to several nodes (a block that
 * is repeated four times has four copies of every layer).
 */
export function idsAtLine(links: CodeLink[], line: number): string[] {
  const hits = links.filter((link) => link.start <= line && line <= link.end);
  if (hits.length === 0) return [];
  const pool = hits.some((link) => link.kind === "node")
    ? hits.filter((link) => link.kind === "node")
    : hits;
  const tightest = Math.min(...pool.map((link) => link.end - link.start));
  return [
    ...new Set(pool.filter((link) => link.end - link.start === tightest).map((link) => link.id)),
  ];
}
