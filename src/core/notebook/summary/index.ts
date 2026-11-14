import { parseKeras2Summaries } from "./keras2";
import { parseKeras3Summaries } from "./keras3";
import { parseTorchinfoSummaries } from "./torchinfo";
import type { ModelSummary } from "./types";

export { extractOutputFacts, type OutputFacts, type ParamMention } from "./facts";
export type { ModelSummary, SummaryDim, SummaryFormat, SummaryRow, SummaryTotals } from "./types";

/**
 * Finds every model summary printed in a cell's outputs, whatever the format.
 * Pass the cell's output texts joined with "\n": Keras 3 splits one summary
 * across several outputs (title, table, totals).
 */
export function parseSummaries(outputText: string): ModelSummary[] {
  if (!/Param #/.test(outputText)) return [];
  return [
    ...parseKeras3Summaries(outputText),
    ...parseKeras2Summaries(outputText),
    ...parseTorchinfoSummaries(outputText),
  ];
}
