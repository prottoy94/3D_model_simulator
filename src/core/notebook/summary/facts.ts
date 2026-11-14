/**
 * Small facts printed in notebook outputs that resolve values static analysis
 * cannot know, such as `num_classes = len(class_names)`.
 */

export type ParamMention = {
  /** Text before the number, e.g. "FireNet". */
  label: string;
  value: number;
  /** True when the print rounded the value ("32.18 M parameters"). */
  approximate: boolean;
};

export type OutputFacts = {
  /** Every class count found, most explicit phrasing first. */
  classCounts: number[];
  paramMentions: ParamMention[];
};

const CLASS_PATTERNS: RegExp[] = [
  // Keras: "Found 633 images belonging to 8 classes."
  /belonging to (\d+) classes/gi,
  // "Number of classes: 8", "Number of micro-organism types (classes) found: 8", "num_classes = 8"
  /(?:number of|num_?)[^\n:=]{0,60}class(?:es)?[^\n:=\d]{0,20}[:=]\s*(\d+)/gi,
  // "8 classes found"
  /\b(\d+) classes\b/gi,
];

const UNIT_SCALE: Record<string, number> = { k: 1e3, m: 1e6, million: 1e6, b: 1e9, billion: 1e9 };

export function extractOutputFacts(text: string): OutputFacts {
  const classCounts: number[] = [];
  for (const pattern of CLASS_PATTERNS) {
    for (const match of text.matchAll(pattern)) {
      const count = Number(match[1]);
      if (count > 0 && count < 100_000 && !classCounts.includes(count)) classCounts.push(count);
    }
  }

  const paramMentions: ParamMention[] = [];
  // "FireNet: 32.18 M parameters", "Trainable parameters: 1,234,567", "25.6M params"
  const withUnit =
    /^(.*?)([\d][\d,]*(?:\.\d+)?)\s*(k|m|b|million|billion)?\s*(?:trainable\s+)?param(?:eter)?s\b/gim;
  for (const match of text.matchAll(withUnit)) {
    const unit = match[3]?.toLowerCase();
    const number = Number(match[2].replace(/,/g, ""));
    if (!Number.isFinite(number)) continue;
    paramMentions.push({
      label: match[1].replace(/[:=|\s]+$/, "").trim(),
      value: Math.round(number * (unit ? UNIT_SCALE[unit] : 1)),
      approximate: Boolean(unit),
    });
  }
  const labelFirst = /^(.*?param(?:eter)?s)\s*[:=]\s*([\d][\d,]*)\s*$/gim;
  for (const match of text.matchAll(labelFirst)) {
    paramMentions.push({
      label: match[1].trim(),
      value: Number(match[2].replace(/,/g, "")),
      approximate: false,
    });
  }
  return { classCounts, paramMentions };
}
