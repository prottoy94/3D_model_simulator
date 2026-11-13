import { NeonButton } from "@/components/ui";

/** Timeline slot. Phase 5 replaces the empty track with step markers and a scrubber. */
export function TimelineBar() {
  return (
    <footer className="glass flex h-14 shrink-0 items-center gap-3 rounded-panel px-3">
      <div className="flex items-center gap-1">
        <NeonButton size="sm" disabled aria-label="Previous step">
          ◀
        </NeonButton>
        <NeonButton size="sm" variant="primary" disabled aria-label="Play">
          ▶
        </NeonButton>
        <NeonButton size="sm" disabled aria-label="Next step">
          ▶▶
        </NeonButton>
      </div>
      <div
        className="relative h-1 flex-1 rounded-full bg-glass-2"
        role="img"
        aria-label="Timeline: no simulation loaded"
      >
        <div className="absolute inset-y-0 left-0 w-0 rounded-full bg-accent" />
      </div>
      <span className="font-mono text-xs tabular-nums text-text-3">step –/–</span>
      <span className="font-mono text-xs tabular-nums text-text-3">1.0×</span>
    </footer>
  );
}
