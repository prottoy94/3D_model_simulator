/**
 * Verification (doc 03 §6): compares what the engine computed with the
 * notebook's own saved `model.summary()` table, row by row and in total.
 */
import type {
  ModelVerification,
  ParamTotals,
  RuntimeGroup,
  RuntimeNode,
  VerificationRow,
} from "../ir/runtime";
import type { ModelSummary, SummaryRow } from "../notebook/summary";
import { layerNames, opForLayerName } from "../ops";
import type { OpKind } from "../ops/kinds";
import { isKnown, shapeText, type TensorShape } from "./tensor";

export function sumParams(nodes: RuntimeNode[]): ParamTotals {
  let trainable = 0;
  let nonTrainable = 0;
  for (const node of nodes) {
    trainable += node.paramCount.trainable;
    nonTrainable += node.paramCount.nonTrainable;
  }
  return { total: trainable + nonTrainable, trainable, nonTrainable };
}

const plain = (text: string) => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** Ops a summary row's layer type may stand for. Truncated names ("MultiHeadAttentio…") match by prefix. */
function opsForRow(row: SummaryRow): Set<OpKind> {
  const ops = new Set<OpKind>();
  if (row.type === "Activation") {
    for (const op of ["relu", "gelu", "sigmoid", "softmax"] as const) ops.add(op);
    return ops;
  }
  const exact = opForLayerName(row.type);
  if (exact) ops.add(exact);
  if (row.truncated && !exact) {
    for (const framework of ["keras", "pytorch"] as const) {
      for (const [name, op] of layerNames(framework)) if (name.startsWith(row.type)) ops.add(op);
    }
  }
  return ops;
}

/** A printed `None` matches anything; a run-time size never matches a printed number. */
function shapeAgrees(row: SummaryRow, computed: TensorShape | null): boolean {
  if (!row.outputShape) return true;
  if (!computed) return false;
  if (row.outputShape.length !== computed.dims.length) return false;
  return row.outputShape.every((printed, i) => {
    if (printed === null || i === 0) return true;
    const dim = computed.dims[i]!;
    return isKnown(dim) && dim === printed;
  });
}

export type VerifyInput = {
  /** The model's nodes in topological order. */
  nodes: RuntimeNode[];
  /** The model's groups. */
  groups: RuntimeGroup[];
  summary: ModelSummary | null;
};

export function verifyModel({ nodes, groups, summary }: VerifyInput): ModelVerification {
  const computed = sumParams(nodes);
  if (!summary) {
    return { totals: { computed, summary: null, match: null }, rows: [], unmatchedRows: [] };
  }

  const used = new Set<string>();
  const usedGroups = new Set<string>();
  const rows: VerificationRow[] = [];
  const unmatchedRows: ModelVerification["unmatchedRows"] = [];

  const nodesInGroup = (groupId: string): RuntimeNode[] => {
    const ids = new Set([groupId]);
    let grew = true;
    while (grew) {
      grew = false;
      for (const group of groups) {
        if (group.parentId && ids.has(group.parentId) && !ids.has(group.id)) {
          ids.add(group.id);
          grew = true;
        }
      }
    }
    return nodes.filter((node) => ids.has(node.groupId));
  };

  for (const row of summary.rows) {
    const ops = opsForRow(row);
    const isWrapper = row.type === "TimeDistributed";
    const node = nodes.find(
      (candidate) =>
        !used.has(candidate.id) &&
        (isWrapper
          ? candidate.wrapper === "time_distributed"
          : candidate.wrapper === null &&
            (ops.has(candidate.op) ||
              (candidate.customOpName !== null && candidate.customOpName === row.type))),
    );

    if (node) {
      used.add(node.id);
      const computedParams = node.paramCount.trainable + node.paramCount.nonTrainable;
      const match =
        (row.params === null || row.params === computedParams) && shapeAgrees(row, node.outShape);
      node.summaryRow = row;
      if (!match && (node.status === "ok" || node.status === "unverified")) {
        node.status = "mismatch";
        node.issues.push({
          code: "summary-mismatch",
          message: `The notebook's summary shows ${row.shapeText || "no shape"} and ${row.params ?? "?"} params for this layer; the engine computed ${node.outShape ? shapeText(node.outShape) : "no shape"} and ${computedParams}.`,
          severity: "warn",
        });
      } else if (
        match &&
        node.status === "unverified" &&
        !node.issues.some((issue) => issue.code !== "declared")
      ) {
        // A declared layer that agrees with the saved summary is as good as computed.
        node.status = "ok";
      }
      rows.push({
        nodeOrGroupId: node.id,
        summaryName: row.name,
        summaryType: row.type,
        summaryShape: row.shapeText || null,
        computedShape: node.outShape ? shapeText(node.outShape) : "?",
        summaryParams: row.params,
        computedParams,
        status: match ? "match" : "mismatch",
      });
      continue;
    }

    // A composite layer (a custom Keras Layer with sub-layers) is one row but a group here.
    const group = groups.find(
      (candidate) =>
        !usedGroups.has(candidate.id) &&
        candidate.parentId !== null &&
        (plain(candidate.label) === plain(row.type) ||
          (row.name !== "" && plain(candidate.label) === plain(row.name))),
    );
    const members = group ? nodesInGroup(group.id).filter((member) => !used.has(member.id)) : [];
    if (group && members.length > 0) {
      usedGroups.add(group.id);
      for (const member of members) used.add(member.id);
      const computedParams = sumParams(members).total;
      const last = members[members.length - 1]!;
      const match =
        (row.params === null || row.params === computedParams) && shapeAgrees(row, last.outShape);
      rows.push({
        nodeOrGroupId: group.id,
        summaryName: row.name,
        summaryType: row.type,
        summaryShape: row.shapeText || null,
        computedShape: last.outShape ? shapeText(last.outShape) : "?",
        summaryParams: row.params,
        computedParams,
        status: match ? "match" : "mismatch",
      });
      continue;
    }

    unmatchedRows.push({ name: row.name, type: row.type, params: row.params });
  }

  const { totals } = summary;
  const agrees = (printed: number | null, value: number) => printed === null || printed === value;
  const match =
    totals.total === null && totals.trainable === null
      ? null
      : agrees(totals.total, computed.total) &&
        agrees(totals.trainable, computed.trainable) &&
        agrees(totals.nonTrainable, computed.nonTrainable);

  return { totals: { computed, summary: totals, match }, rows, unmatchedRows };
}
