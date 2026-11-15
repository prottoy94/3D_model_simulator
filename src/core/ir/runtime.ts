/** Runtime IR (doc 03 §2): typed, unrolled, shaped and verified. Views read only this. */
import type { ModelSummary, SummaryRow } from "../notebook/summary";
import type { ParamCount } from "../ops/types";
import type { Framework, TensorShape } from "../shapes/tensor";
import type { CodeRef, WireEdge, WireGroup, WireIR, WireNode } from "./wire";

export type IssueSeverity = "info" | "warn" | "error";

export type Issue = { code: string; message: string; severity: IssueSeverity };

/**
 * ok: computed by the engine (and equal to the saved summary when there is one).
 * mismatch: computed, but the saved summary disagrees.
 * unverified: declared or downstream of a problem; shown but not trusted.
 * error: the engine could not compute this node.
 */
export type NodeStatus = "ok" | "mismatch" | "unverified" | "error";

export type RuntimeNode = Omit<WireNode, "params"> & {
  /** Typed by the op's catalog schema, with framework defaults applied. */
  params: Record<string, unknown>;
  wrapper: "time_distributed" | null;
  inShapes: TensorShape[];
  /** null when inference failed and there was no claimed shape to fall back on. */
  outShape: TensorShape | null;
  paramCount: ParamCount;
  /** Multiply-adds × 2, per sample. */
  flops: number;
  status: NodeStatus;
  issues: Issue[];
  provenance: "ai" | "zoo" | "ai-synth" | "heuristic" | "user";
  /** Canonical node id when this node is an unrolled repeat instance. */
  templateId: string | null;
  summaryRow: SummaryRow | null;
  /** Names of the symbols this node's params were written with (e.g. "PATCH_SIZE"). Lens uses them. */
  symbolsUsed: string[];
};

export type RuntimeGroup = WireGroup & {
  /** Canonical group id when this group is an unrolled repeat instance. */
  templateId: string | null;
  stats: { params: ParamCount; flops: number; nodeCount: number; depth: number };
  expandedByDefault: boolean;
};

export type ParamTotals = { total: number; trainable: number; nonTrainable: number };

export type VerificationRow = {
  nodeOrGroupId: string;
  summaryName: string;
  summaryType: string;
  summaryShape: string | null;
  computedShape: string;
  summaryParams: number | null;
  computedParams: number;
  status: "match" | "mismatch";
};

export type ModelVerification = {
  totals: {
    computed: ParamTotals;
    summary: { total: number | null; trainable: number | null; nonTrainable: number | null } | null;
    /** null when there is no saved summary to compare against. */
    match: boolean | null;
  };
  rows: VerificationRow[];
  /** Summary rows the engine could not pair with a node or group. */
  unmatchedRows: Array<{ name: string; type: string; params: number | null }>;
};

export type VerificationReport = {
  perModel: Record<string, ModelVerification>;
  llmDisagreements: Array<{ nodeId: string; claimed: string; computed: string }>;
  repairAttempts: number;
};

export type CodeInterval = CodeRef & { nodeId?: string; groupId?: string };

export type RuntimeIR = Omit<WireIR, "nodes" | "groups"> & {
  engineFramework: Framework;
  nodes: Record<string, RuntimeNode>;
  groups: Record<string, RuntimeGroup>;
  index: {
    topoOrder: Record<string, string[]>;
    children: Record<string, { groups: string[]; nodes: string[] }>;
    inEdges: Record<string, WireEdge[]>;
    outEdges: Record<string, WireEdge[]>;
    /** Sorted by cell, then start line. Lens builds its interval lookup from this. */
    codeIntervals: CodeInterval[];
  };
  /** Problems that are not tied to one node (dangling edges, cycles). */
  issues: Issue[];
  verification: VerificationReport;
  assumptions: Array<{ symbol: string; value: string; source: string; editable: true }>;
};

export type BuildOptions = {
  /** Saved `model.summary()` tables from the notebook, matched to models by name. */
  summaries?: ModelSummary[];
  provenance?: RuntimeNode["provenance"];
};
