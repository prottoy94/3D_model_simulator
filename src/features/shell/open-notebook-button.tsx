"use client";

import { useRef } from "react";
import { NeonButton } from "@/components/ui";
import { useProjectStore } from "@/stores/project-store";

type OpenNotebookButtonProps = {
  label?: string;
  variant?: "primary" | "ghost";
  size?: "sm" | "md";
  /** Called after a notebook was loaded successfully. */
  onOpened?: () => void;
};

/** A button that opens the file picker and loads the chosen notebook. */
export function OpenNotebookButton({
  label = "Open notebook",
  variant = "ghost",
  size = "sm",
  onOpened,
}: OpenNotebookButtonProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const openFile = useProjectStore((state) => state.openFile);
  const loading = useProjectStore((state) => state.status === "loading");

  return (
    <>
      <NeonButton
        variant={variant}
        size={size}
        disabled={loading}
        onClick={() => inputRef.current?.click()}
      >
        {loading ? "Reading…" : label}
      </NeonButton>
      <input
        ref={inputRef}
        type="file"
        accept=".ipynb"
        className="sr-only"
        aria-label="Choose a notebook file"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          event.target.value = ""; // allow choosing the same file again
          if (file && (await openFile(file))) onOpened?.();
        }}
      />
    </>
  );
}
