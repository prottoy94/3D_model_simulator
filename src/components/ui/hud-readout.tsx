import { cn } from "@/lib/cn";

/** Where a displayed number came from (golden rule 8: honest instruments). */
export type ReadoutSource = "summary" | "computed" | "mismatch" | "placeholder";

const sourceMark: Record<ReadoutSource, { glyph: string; label: string; className: string }> = {
  summary: { glyph: "✓", label: "matches notebook summary", className: "text-ok" },
  computed: { glyph: "≈", label: "computed", className: "text-text-3" },
  mismatch: { glyph: "⚠", label: "mismatch", className: "text-error" },
  placeholder: { glyph: "·", label: "no data yet", className: "text-text-3" },
};

type HudReadoutProps = {
  label: string;
  value: string;
  source?: ReadoutSource;
  className?: string;
};

export function HudReadout({ label, value, source, className }: HudReadoutProps) {
  const mark = source ? sourceMark[source] : null;
  return (
    <div className={cn("flex items-baseline justify-between gap-3 py-1", className)}>
      <span className="text-xs text-text-3">{label}</span>
      <span className="flex items-baseline gap-1.5 font-mono text-[13px] tabular-nums text-text-1">
        {value}
        {mark ? (
          <span className={mark.className} title={mark.label} aria-label={mark.label}>
            {mark.glyph}
          </span>
        ) : null}
      </span>
    </div>
  );
}
