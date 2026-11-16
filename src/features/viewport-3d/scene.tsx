"use client";

import { Grid, OrbitControls, PerformanceMonitor, Sparkles } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { useMemo, useState, type RefObject } from "react";
import { MOUSE, TOUCH } from "three";
import type { ModelLayout } from "@/core/layout";
import { readToken } from "@/lib/tokens";
import { usePrefsStore } from "@/stores/prefs-store";
import type { LabelRegistry } from "./labels";
import { ModelScene } from "./model-scene";

// Stable references: a new object on each render would make the controls reset themselves.
const INITIAL_TARGET: [number, number, number] = [0, 1, 0];
const MOUSE_BUTTONS = { LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE };
const TOUCHES = { ONE: TOUCH.PAN, TWO: TOUCH.DOLLY_ROTATE };

/**
 * The specimen stage: grid floor, fog and drifting dust, with the model's
 * architecture (glyphs, edges, group frames) on it when there is one.
 */
export default function Scene({
  layout,
  labels,
}: {
  layout: ModelLayout | null;
  labels: RefObject<LabelRegistry>;
}) {
  // Drop to 1× pixel density when the frame rate sags (large models, weak GPUs).
  const [maxDpr, setMaxDpr] = useState(2);
  // Bloom adds light, which washes out a pale stage: it is for the dark theme only.
  const light = usePrefsStore((state) => state.theme === "light");
  const colors = useMemo(
    () => ({
      deep: readToken("--bg-deep"),
      accent: readToken("--accent"),
      accent2: readToken("--accent-2"),
      grid: readToken("--op-shape"),
    }),
    [],
  );

  return (
    <Canvas
      dpr={[1, maxDpr]}
      camera={{ position: [9, 6, 12], fov: 42, near: 0.1, far: 400 }}
      // The drawing buffer is kept so the view can be saved as a picture at any time.
      gl={{ antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true }}
    >
      <color attach="background" args={[colors.deep]} />
      <fogExp2 attach="fog" args={[colors.deep, 0.011]} />
      <PerformanceMonitor onDecline={() => setMaxDpr(1)} />
      <ambientLight intensity={0.4} />
      <directionalLight position={[6, 10, 4]} intensity={0.8} />

      <Grid
        infiniteGrid
        cellSize={1}
        sectionSize={5}
        cellThickness={0.45}
        sectionThickness={0.7}
        cellColor={colors.grid}
        sectionColor={colors.accent}
        fadeDistance={90}
        fadeStrength={2.2}
        position={[0, -0.01, 0]}
      />
      <Sparkles count={140} scale={[36, 9, 36]} size={2.2} speed={0.25} color={colors.accent} />
      <Sparkles count={50} scale={[36, 9, 36]} size={3} speed={0.15} color={colors.accent2} />

      {layout ? <ModelScene layout={layout} labels={labels} /> : null}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={2}
        maxDistance={160}
        maxPolarAngle={Math.PI / 2.05}
        target={INITIAL_TARGET}
        // Dragging moves the view, like a map: models are long, so panning is the main gesture.
        // Right-drag (or Ctrl/Shift + drag) rotates and the wheel zooms.
        mouseButtons={MOUSE_BUTTONS}
        touches={TOUCHES}
        screenSpacePanning
        panSpeed={1.1}
      />

      {light ? null : (
        <EffectComposer>
          <Bloom intensity={0.9} luminanceThreshold={0.25} luminanceSmoothing={0.4} mipmapBlur />
        </EffectComposer>
      )}
    </Canvas>
  );
}
