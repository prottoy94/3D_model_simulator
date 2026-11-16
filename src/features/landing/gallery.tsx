"use client";

import { useRouter } from "next/navigation";
import { Chip, GlassPanel } from "@/components/ui";
import { useModelStore } from "@/stores/model-store";

type GalleryModel = {
  name: string;
  family: string;
  color: string;
  example?: string;
  note?: string;
};

const gallery: ReadonlyArray<GalleryModel> = [
  {
    name: "AlexNet",
    family: "CNN",
    color: "var(--op-conv)",
    example: "alexnet",
    note: "17 layers · 46.8 M params",
  },
  {
    name: "VGG16",
    family: "CNN",
    color: "var(--op-conv)",
    example: "vgg16",
    note: "23 layers · 15.0 M params",
  },
  {
    name: "Mini Vision Transformer",
    family: "Attention",
    color: "var(--op-attn)",
    example: "mini-vit",
    note: "4 blocks · 404 K params",
  },
  {
    name: "FireNet",
    family: "Detection · attention",
    color: "var(--op-det)",
    example: "firenet",
    note: "From your notebook · 32.18 M params",
  },
  { name: "ResNet50", family: "CNN · residual", color: "var(--op-merge)" },
  { name: "EfficientNetB0", family: "CNN · MBConv", color: "var(--op-pool)" },
  { name: "LSTM", family: "Recurrent", color: "var(--op-recur)" },
  { name: "Faster R-CNN", family: "Detection", color: "var(--op-det)" },
  { name: "U-Net", family: "Segmentation", color: "var(--op-norm)" },
];

/** Prebuilt architectures. The first three open in Scope; the rest are planned. */
export function Gallery() {
  const router = useRouter();
  const open = (example: string) => {
    if (useModelStore.getState().loadExample(example)) router.push("/scope");
  };

  return (
    <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {gallery.map((model) => {
        const card = (
          <GlassPanel
            className={`h-full gap-2 p-4 text-left transition-[border-color] duration-150 ${
              model.example ? "group-hover:border-accent/60" : "opacity-50"
            }`}
          >
            <span className="font-display text-sm font-semibold text-text-1">{model.name}</span>
            <Chip color={model.color}>{model.family}</Chip>
            <span className="text-[11px] text-text-3">{model.note ?? "Planned"}</span>
          </GlassPanel>
        );
        return (
          <li key={model.name}>
            {model.example ? (
              <button
                type="button"
                className="group block h-full w-full"
                onClick={() => open(model.example!)}
              >
                {card}
              </button>
            ) : (
              card
            )}
          </li>
        );
      })}
    </ul>
  );
}
