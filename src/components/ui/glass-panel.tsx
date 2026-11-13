import type { ComponentProps, ReactNode } from "react";
import { cn } from "@/lib/cn";

type GlassPanelProps = ComponentProps<"section"> & {
  /** Raised panels (popovers, drawers) use the more opaque glass. */
  raised?: boolean;
};

export function GlassPanel({ raised = false, className, children, ...rest }: GlassPanelProps) {
  return (
    <section
      className={cn(
        "glass relative flex min-h-0 flex-col overflow-hidden rounded-panel",
        raised && "bg-glass-2",
        className,
      )}
      {...rest}
    >
      {/* 1px gradient highlight on the top-left edge, the "holographic" rim. */}
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0 rounded-panel opacity-60"
        style={{
          padding: 1,
          background: "var(--glass-highlight)",
          mask: "linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0)",
          maskComposite: "exclude",
          WebkitMaskComposite: "xor",
        }}
      />
      {children}
    </section>
  );
}

type PanelHeaderProps = {
  title: string;
  actions?: ReactNode;
  className?: string;
};

export function PanelHeader({ title, actions, className }: PanelHeaderProps) {
  return (
    <header
      className={cn(
        "flex h-9 shrink-0 items-center justify-between border-b border-line px-3",
        className,
      )}
    >
      <h2 className="hud-label">{title}</h2>
      {actions ? <div className="flex items-center gap-1">{actions}</div> : null}
    </header>
  );
}
