"use client";

import { Grid, OrbitControls, Sparkles } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { useMemo } from "react";
import { readToken } from "@/lib/tokens";

/**
 * The empty specimen stage: grid floor, fog and drifting dust.
 * Phase 3 mounts the architecture (glyphs, edges, groups) inside this canvas.
 */
export default function Scene() {
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
      dpr={[1, 2]}
      camera={{ position: [9, 6, 12], fov: 42, near: 0.1, far: 200 }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
    >
      <color attach="background" args={[colors.deep]} />
      <fogExp2 attach="fog" args={[colors.deep, 0.035]} />
      <ambientLight intensity={0.4} />
      <directionalLight position={[6, 10, 4]} intensity={0.8} />

      <Grid
        infiniteGrid
        cellSize={1}
        sectionSize={5}
        cellThickness={0.6}
        sectionThickness={0.8}
        cellColor={colors.grid}
        sectionColor={colors.accent}
        fadeDistance={55}
        fadeStrength={2.2}
        position={[0, -0.01, 0]}
      />
      <Sparkles count={140} scale={[36, 9, 36]} size={2.2} speed={0.25} color={colors.accent} />
      <Sparkles count={50} scale={[36, 9, 36]} size={3} speed={0.15} color={colors.accent2} />

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={4}
        maxDistance={60}
        maxPolarAngle={Math.PI / 2.05}
        target={[0, 1, 0]}
      />

      <EffectComposer>
        <Bloom intensity={0.9} luminanceThreshold={0.25} luminanceSmoothing={0.4} mipmapBlur />
      </EffectComposer>
    </Canvas>
  );
}
