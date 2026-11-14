import Link from "next/link";
import { Chip, GlassPanel } from "@/components/ui";
import { Constellation } from "@/features/landing/constellation";
import { DropZone } from "@/features/landing/drop-zone";

const gallery: ReadonlyArray<{ name: string; family: string; color: string }> = [
  { name: "AlexNet", family: "CNN", color: "var(--op-conv)" },
  { name: "VGG16", family: "CNN", color: "var(--op-conv)" },
  { name: "ResNet50", family: "CNN · residual", color: "var(--op-merge)" },
  { name: "EfficientNetB0", family: "CNN · MBConv", color: "var(--op-pool)" },
  { name: "Vision Transformer", family: "Attention", color: "var(--op-attn)" },
  { name: "LSTM", family: "Recurrent", color: "var(--op-recur)" },
  { name: "Faster R-CNN", family: "Detection", color: "var(--op-det)" },
  { name: "U-Net", family: "Segmentation", color: "var(--op-norm)" },
];

export default function LandingPage() {
  return (
    <div className="relative flex min-h-dvh flex-col overflow-hidden">
      <Constellation />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 70% 55% at 50% 35%, transparent, var(--bg-void) 85%)",
        }}
      />

      <header className="relative z-10 flex h-14 items-center justify-between px-6">
        <span className="flex items-center gap-2">
          <span aria-hidden className="text-2xl leading-none text-accent">
            ◈
          </span>
          <span className="font-brand text-[13px] font-bold tracking-[0.28em]">NEUROSCOPE</span>
        </span>
        <nav className="flex items-center gap-5 text-sm text-text-2">
          <Link href="/scope" className="transition-colors hover:text-text-1">
            Scope
          </Link>
          <Link href="/forge" className="transition-colors hover:text-text-1">
            Forge
          </Link>
        </nav>
      </header>

      <main className="relative z-10 flex flex-1 flex-col items-center px-4 pb-16 pt-[8vh]">
        <p className="hud-label !text-accent">3D model simulator</p>
        <h1 className="mt-3 max-w-3xl text-balance text-center font-display text-4xl font-bold uppercase leading-tight tracking-[0.04em] text-text-1 sm:text-6xl">
          Watch your model think.
        </h1>
        <p className="mt-4 max-w-xl text-pretty text-center text-base leading-relaxed text-text-2">
          Turn any Jupyter notebook into a live 3D simulation of its architecture. Step through
          every layer, and highlight code to see exactly what it does to a tensor.
        </p>

        <div className="mt-10 flex w-full justify-center">
          <DropZone />
        </div>

        <section className="mt-16 w-full max-w-5xl" aria-labelledby="gallery-title">
          <div className="mb-3 flex items-baseline justify-between">
            <h2 id="gallery-title" className="hud-label">
              Model gallery
            </h2>
            <span className="text-xs text-text-3">Prebuilt models arrive in Phase 2</span>
          </div>
          <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {gallery.map((model) => (
              <li key={model.name}>
                <GlassPanel className="gap-3 p-4 opacity-70">
                  <span className="font-display text-sm font-semibold text-text-1">
                    {model.name}
                  </span>
                  <Chip color={model.color}>{model.family}</Chip>
                </GlassPanel>
              </li>
            ))}
          </ul>
        </section>
      </main>
    </div>
  );
}
