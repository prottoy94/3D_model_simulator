import type { ComponentProps } from "react";
import { cn } from "@/lib/cn";

type Variant = "primary" | "ghost" | "danger";
type Size = "sm" | "md";

type NeonButtonProps = ComponentProps<"button"> & {
  variant?: Variant;
  size?: Size;
};

const variants: Record<Variant, string> = {
  primary:
    "border-accent/60 bg-accent/10 text-accent hover:bg-accent/20 hover:shadow-[0_0_24px_rgba(34,211,238,0.3)]",
  ghost: "border-line bg-transparent text-text-2 hover:border-line-strong hover:text-text-1",
  danger: "border-error/60 bg-error/10 text-error hover:bg-error/20",
};

const sizes: Record<Size, string> = {
  sm: "h-8 px-2.5 text-xs",
  md: "h-9 px-4 text-[13px]",
};

export function NeonButton({
  variant = "ghost",
  size = "md",
  className,
  type = "button",
  ...rest
}: NeonButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-control border font-medium transition-[background,box-shadow,color,border-color] duration-150",
        "disabled:pointer-events-none disabled:opacity-40",
        variants[variant],
        sizes[size],
        className,
      )}
      {...rest}
    />
  );
}
