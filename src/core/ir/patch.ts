/**
 * RepairPatch (doc 03 §6): a small list of edits an LLM sends back after the
 * engine reports mismatches. Applying it never mutates the input IR.
 */
import { z } from "zod";
import { wireEdgeSchema, wireGroupSchema, wireNodeSchema, type WireIR } from "./wire";

export const repairPatchSchema = z.strictObject({
  ops: z.array(
    z.strictObject({
      action: z.enum([
        "update_node",
        "add_node",
        "remove_node",
        "add_edge",
        "remove_edge",
        "update_group",
        "set_symbol",
      ]),
      targetId: z.string(),
      node: wireNodeSchema.nullable(),
      edge: wireEdgeSchema.nullable(),
      group: wireGroupSchema.nullable(),
      symbol: z.strictObject({ name: z.string(), value: z.string() }).nullable(),
      reason: z.string(),
    }),
  ),
});

export type RepairPatch = z.infer<typeof repairPatchSchema>;

export type PatchResult = { ir: WireIR; applied: number; errors: string[] };

export function applyRepairPatch(wire: WireIR, patch: RepairPatch): PatchResult {
  const ir = structuredClone(wire);
  const errors: string[] = [];
  let applied = 0;

  for (const [position, op] of patch.ops.entries()) {
    const fail = (message: string) =>
      errors.push(`op ${position} (${op.action} ${op.targetId}): ${message}`);
    switch (op.action) {
      case "update_node": {
        const index = ir.nodes.findIndex((node) => node.id === op.targetId);
        if (index < 0 || !op.node) fail(index < 0 ? "node not found" : "no node given");
        else if (op.node.id !== op.targetId) fail("a node's id cannot change");
        else {
          ir.nodes[index] = op.node;
          applied++;
        }
        break;
      }
      case "add_node": {
        if (!op.node) fail("no node given");
        else if (ir.nodes.some((node) => node.id === op.node!.id))
          fail("a node with this id already exists");
        else {
          ir.nodes.push(op.node);
          applied++;
        }
        break;
      }
      case "remove_node": {
        const before = ir.nodes.length;
        ir.nodes = ir.nodes.filter((node) => node.id !== op.targetId);
        if (ir.nodes.length === before) fail("node not found");
        else {
          ir.edges = ir.edges.filter(
            (edge) => edge.from !== op.targetId && edge.to !== op.targetId,
          );
          for (const model of ir.models) {
            model.inputNodeIds = model.inputNodeIds.filter((id) => id !== op.targetId);
            model.outputNodeIds = model.outputNodeIds.filter((id) => id !== op.targetId);
          }
          applied++;
        }
        break;
      }
      case "add_edge": {
        if (!op.edge) fail("no edge given");
        else if (ir.edges.some((edge) => edge.id === op.edge!.id))
          fail("an edge with this id already exists");
        else {
          ir.edges.push(op.edge);
          applied++;
        }
        break;
      }
      case "remove_edge": {
        const before = ir.edges.length;
        ir.edges = ir.edges.filter((edge) => edge.id !== op.targetId);
        if (ir.edges.length === before) fail("edge not found");
        else applied++;
        break;
      }
      case "update_group": {
        const index = ir.groups.findIndex((group) => group.id === op.targetId);
        if (index < 0 || !op.group) fail(index < 0 ? "group not found" : "no group given");
        else if (op.group.id !== op.targetId) fail("a group's id cannot change");
        else {
          ir.groups[index] = op.group;
          applied++;
        }
        break;
      }
      case "set_symbol": {
        if (!op.symbol) {
          fail("no symbol given");
          break;
        }
        const existing = ir.symbols.find((symbol) => symbol.name === op.symbol!.name);
        if (existing) existing.value = op.symbol.value;
        else {
          ir.symbols.push({
            name: op.symbol.name,
            value: op.symbol.value,
            provenance: "assumed",
            ref: null,
            note: op.reason,
          });
        }
        applied++;
        break;
      }
    }
  }
  return { ir, applied, errors };
}
