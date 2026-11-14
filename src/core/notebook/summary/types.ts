export type SummaryFormat = "keras3" | "keras2" | "torchinfo";

/** One dimension as printed: a number, or null for `None` / unknown. */
export type SummaryDim = number | null;

export type SummaryRow = {
  /** Layer instance name, e.g. "conv2d_1". Empty when the format does not print one. */
  name: string;
  /** Layer class, e.g. "Conv2D". */
  type: string;
  /** True when the table cut the name or type short with "…". Treat them as prefixes. */
  truncated: boolean;
  outputShape: SummaryDim[] | null;
  /** The shape exactly as printed, for display. */
  shapeText: string;
  params: number | null;
  /** Keras functional summaries: names of the inbound layers (possibly truncated). */
  connectedTo: string[];
  /** torchinfo nesting depth (1 = top-level child). 0 when the format is flat. */
  depth: number;
};

export type SummaryTotals = {
  total: number | null;
  trainable: number | null;
  nonTrainable: number | null;
};

export type ModelSummary = {
  format: SummaryFormat;
  modelName: string | null;
  rows: SummaryRow[];
  totals: SummaryTotals;
};
