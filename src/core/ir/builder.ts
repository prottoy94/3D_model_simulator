/**
 * A small helper for writing Wire IR by hand or from a parser, without
 * repeating every required field. Used by the fast-path parser, the IR
 * fixtures and tests. It only builds data; validation happens in normalize.
 */
import type { OpKind } from "../ops/kinds";
import {
  IR_VERSION,
  type CodeRef,
  type ParamValue,
  type WireEdge,
  type WireGroup,
  type WireIR,
  type WireNode,
  type WireSymbol,
} from "./wire";

type Meta = Pick<WireIR, "title" | "summary" | "framework"> &
  Partial<Pick<WireIR, "scope" | "domain" | "task">>;

type NodeOptions = {
  /** Inputs in port order. Use `{ id, kind }` for a residual or skip connection. */
  from?: Array<string | { id: string; kind: WireEdge["kind"] }>;
  group?: string;
  label?: string;
  refs?: CodeRef[];
  explanation?: string;
  customOpName?: string;
  trainable?: WireNode["trainable"];
  claimed?: Array<number | string>;
  confidence?: WireNode["confidence"];
};

type GroupOptions = Partial<
  Pick<WireGroup, "kind" | "repeatCount" | "repeatSymbol" | "refs" | "description" | "zooKey">
> & {
  parent?: string;
};

export const ref = (
  cell: number,
  startLine: number,
  endLine: number,
  role: CodeRef["role"] = "declare",
): CodeRef => ({
  cell,
  startLine,
  endLine,
  role,
});

export class WireBuilder {
  private readonly ir: WireIR;
  private modelId = "";
  private groupId = "";
  private edgeCount = 0;

  constructor(meta: Meta) {
    this.ir = {
      irVersion: IR_VERSION,
      scope: meta.scope ?? "notebook",
      title: meta.title,
      summary: meta.summary,
      framework: meta.framework,
      domain: meta.domain ?? "cv",
      task: meta.task ?? "classification",
      symbols: [],
      models: [],
      groups: [],
      nodes: [],
      edges: [],
      pipelines: [],
      training: null,
      warnings: [],
    };
  }

  symbol(
    name: string,
    value: unknown,
    extra: Partial<Omit<WireSymbol, "name" | "value">> = {},
  ): this {
    this.ir.symbols.push({
      name,
      value: JSON.stringify(value),
      provenance: extra.provenance ?? "literal",
      ref: extra.ref ?? null,
      note: extra.note ?? "",
    });
    return this;
  }

  /** Starts a model and its root group. Later nodes and groups go into it. */
  model(id: string, name: string, description = "", refs: CodeRef[] = []): this {
    const rootGroupId = `g_${id.replace(/^m_/, "")}`;
    this.ir.models.push({
      id,
      name,
      description,
      rootGroupId,
      inputNodeIds: [],
      outputNodeIds: [],
      isPrimary: this.ir.models.length === 0,
      variantOf: null,
    });
    this.modelId = id;
    this.pushGroup(rootGroupId, name, { kind: "model", refs, description }, null);
    this.groupId = rootGroupId;
    return this;
  }

  /** Adds a group and makes it the current one. Its parent defaults to the model's root group. */
  group(id: string, label: string, options: GroupOptions = {}): this {
    const model = this.ir.models.find((candidate) => candidate.id === this.modelId);
    this.pushGroup(id, label, options, options.parent ?? model?.rootGroupId ?? null);
    this.groupId = id;
    return this;
  }

  /** Makes an existing group current again (use the root group id to leave a block). */
  into(groupId: string): this {
    this.groupId = groupId;
    return this;
  }

  node(
    id: string,
    op: OpKind,
    params: Record<string, ParamValue> = {},
    options: NodeOptions = {},
  ): this {
    this.ir.nodes.push({
      id,
      groupId: options.group ?? this.groupId,
      op,
      customOpName: options.customOpName ?? null,
      label: options.label ?? id.replace(/^n_/, ""),
      params: Object.entries(params).map(([key, value]) => ({ key, value })),
      trainable: options.trainable ?? "yes",
      claimedOutputShape: options.claimed ?? null,
      refs: options.refs ?? [],
      explanation: options.explanation ?? "",
      confidence: options.confidence ?? "high",
    });
    (options.from ?? []).forEach((source, toPort) => {
      const from = typeof source === "string" ? source : source.id;
      const kind = typeof source === "string" ? "data" : source.kind;
      this.ir.edges.push({ id: `e${++this.edgeCount}`, from, to: id, toPort, kind });
    });
    const model = this.ir.models.find((candidate) => candidate.id === this.modelId);
    if (model && op === "input") model.inputNodeIds.push(id);
    return this;
  }

  /** Marks the current model's output nodes. */
  outputs(...nodeIds: string[]): this {
    const model = this.ir.models.find((candidate) => candidate.id === this.modelId);
    if (model) model.outputNodeIds = nodeIds;
    return this;
  }

  training(training: WireIR["training"]): this {
    this.ir.training = training;
    return this;
  }

  warn(message: string): this {
    this.ir.warnings.push(message);
    return this;
  }

  build(): WireIR {
    return structuredClone(this.ir);
  }

  private pushGroup(
    id: string,
    label: string,
    options: GroupOptions,
    parentId: string | null,
  ): void {
    this.ir.groups.push({
      id,
      parentId,
      modelId: this.modelId,
      label,
      kind: options.kind ?? "block",
      repeatCount: options.repeatCount ?? null,
      repeatSymbol: options.repeatSymbol ?? null,
      zooKey: options.zooKey ?? null,
      zooArgs: [],
      refs: options.refs ?? [],
      description: options.description ?? "",
    });
  }
}
