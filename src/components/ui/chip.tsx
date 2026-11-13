import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

type ChipProps = {
  children: ReactNode;
  /** Any CSS color, normally a token such as `var(--op-conv)`. */
  color?: string;
  className?: string;
};

/** Small pill for op families, statuses and provenance. The dot carries the color. */
export function Chip({ children, color = "var(--text-3)", className }: ChipProps) {
  return (
    <span
      className={cn(
        "inline-flex h-6 items-center gap-1.5 rounded-full border border-line bg-glass-2 px-2.5 text-xs text-text-2",
        className,
      )}
    >
      <span
        aria-hidden
        className="size-1.5 rounded-full"
        style={{ background: color, boxShadow: `0 0 8px ${color}` }}
      />
      {children}
    </span>
  );
}
