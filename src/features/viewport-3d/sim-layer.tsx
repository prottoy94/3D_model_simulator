"use client";

import { useFrame } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import {
  bezierPoint,
  type LayoutEdge,
  type LayoutItem,
  type ModelLayout,
  type Vec3,
} from "@/core/layout";
import { FAMILY_COLOR_TOKEN } from "@/core/ops";
import { stateAt } from "@/core/sim/plan";
import type { Picture, TensorView } from "@/core/sim/view";
import { readToken } from "@/lib/tokens";
import { simClock, useSimStore } from "@/stores/sim-store";

const MAX_NEURONS = 18;
const MAX_PULSES = 8;

/** The picture of a layer's output, on the face of its block that looks back toward the input. */
function FeaturePlane({ item, picture }: { item: LayoutItem; picture: Picture }) {
  const texture = useMemo(() => {
    // WebGL textures start at the bottom row, pictures at the top one.
    const { width, height, rgba } = picture;
    const flipped = new Uint8Array(rgba.length);
    for (let y = 0; y < height; y++) {
      flipped.set(rgba.subarray(y * width * 4, (y + 1) * width * 4), (height - 1 - y) * width * 4);
    }
    const result = new THREE.DataTexture(flipped, width, height, THREE.RGBAFormat);
    result.magFilter = THREE.NearestFilter;
    result.minFilter = THREE.LinearFilter;
    result.colorSpace = THREE.SRGBColorSpace;
    result.needsUpdate = true;
    return result;
  }, [picture]);
  useEffect(() => () => texture.dispose(), [texture]);

  return (
    <mesh
      position={[item.position[0] - item.size[0] / 2 - 0.012, item.position[1], item.position[2]]}
      rotation={[0, -Math.PI / 2, 0]}
      raycast={() => null}
    >
      <planeGeometry args={[item.size[2], item.size[1]]} />
      {/* Slightly dimmed so the bloom pass does not wash the picture out. */}
      <meshBasicMaterial
        map={texture}
        color={[0.85, 0.85, 0.85]}
        side={THREE.DoubleSide}
        toneMapped={false}
      />
    </mesh>
  );
}

/** Neuron columns: each dot's brightness is the size of that unit's value. */
function ActiveNeurons({
  entries,
}: {
  entries: Array<{ item: LayoutItem; values: Float32Array }>;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const dots = useMemo(() => {
    const result: Array<{ position: Vec3; radius: number; level: number }> = [];
    for (const { item, values } of entries) {
      const count = Math.max(1, Math.min(MAX_NEURONS, item.units ?? values.length));
      const step = item.size[1] / count;
      let peak = 1e-9;
      for (const value of values) peak = Math.max(peak, Math.abs(value));
      for (let i = 0; i < count; i++) {
        const value = values[Math.floor((i / count) * values.length)] ?? 0;
        result.push({
          position: [
            item.position[0],
            item.position[1] - item.size[1] / 2 + step * (i + 0.5),
            item.position[2],
          ],
          radius: Math.min(0.1, step * 0.4),
          level: Math.abs(value) / peak,
        });
      }
    }
    return result;
  }, [entries]);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    dots.forEach((dot, index) => {
      dummy.position.set(...dot.position);
      dummy.scale.setScalar(dot.radius);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
      mesh.setColorAt(index, color.setScalar(0.12 + 1.1 * dot.level));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [dots]);

  if (dots.length === 0) return null;
  return (
    <instancedMesh
      key={dots.length}
      ref={ref}
      args={[undefined, undefined, dots.length]}
      raycast={() => null}
    >
      <sphereGeometry args={[1, 12, 10]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

type Controls = { target: THREE.Vector3; update(): void };

/**
 * The forward pass on top of the static model: pulses travel along the edges
 * into the running block, the block glows while it works, and its output
 * picture stays on it afterwards. Everything per-frame goes through refs.
 */
export function SimLayer({ layout }: { layout: ModelLayout }) {
  const plan = useSimStore((state) => state.plan);
  const step = useSimStore((state) => state.step);
  const ended = useSimStore((state) => state.ended);
  const views = useSimStore((state) => state.views);
  const started = useSimStore((state) => state.status !== "idle");

  const glow = useRef<THREE.LineSegments>(null);
  const pulses = useRef<Array<THREE.Mesh | null>>([]);
  const activePlane = useRef<THREE.Group>(null);

  const items = useMemo(() => new Map(layout.items.map((item) => [item.id, item])), [layout]);
  const active = started && plan ? items.get(plan.steps[step]?.itemId ?? "") : undefined;
  const incoming = useMemo<LayoutEdge[]>(
    () => (active ? layout.edges.filter((edge) => edge.to === active.id).slice(0, MAX_PULSES) : []),
    [layout, active],
  );

  /** The view of a block: its last layer's output (a collapsed group shows what leaves it). */
  const viewOf = (item: LayoutItem): TensorView | undefined =>
    views[item.memberIds[item.memberIds.length - 1]!];

  // Blocks the pass has already left. They change once per step, so React handles them.
  const done = useMemo(() => {
    if (!started || !plan) return [];
    return plan.steps
      .filter((entry) => ended || entry.index < step)
      .map((entry) => items.get(entry.itemId))
      .filter((item): item is LayoutItem => item !== undefined);
  }, [started, plan, step, ended, items]);

  const columns = done.flatMap((item) => {
    const values = item.glyph === "neuron-column" ? viewOf(item)?.values : null;
    return values ? [{ item, values }] : [];
  });
  const activeView = active ? viewOf(active) : undefined;
  const palette = useMemo(
    () => ({
      accent: new THREE.Color(readToken("--accent")),
      white: new THREE.Color(readToken("--text-1")),
    }),
    [],
  );
  const activeColor = useMemo(
    () =>
      active
        ? new THREE.Color(readToken(FAMILY_COLOR_TOKEN[active.family] as `--${string}`))
        : palette.accent,
    [active, palette],
  );

  const glowGeometry = useMemo(() => {
    if (!active) return null;
    const grow = 0.06;
    const min = active.position.map((value, axis) => value - active.size[axis]! / 2 - grow) as Vec3;
    const max = active.position.map((value, axis) => value + active.size[axis]! / 2 + grow) as Vec3;
    return new THREE.BoxGeometry(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  }, [active]);
  const glowEdges = useMemo(
    () => (glowGeometry ? new THREE.EdgesGeometry(glowGeometry) : null),
    [glowGeometry],
  );
  useEffect(
    () => () => {
      glowGeometry?.dispose();
      glowEdges?.dispose();
    },
    [glowGeometry, glowEdges],
  );

  useFrame((frame, delta) => {
    if (!plan || !active) return;
    const state = stateAt(plan, simClock.time);
    const sim = useSimStore.getState();

    // Data arriving: one pulse per incoming edge, from producer to this block.
    incoming.forEach((edge, index) => {
      const mesh = pulses.current[index];
      if (!mesh) return;
      mesh.visible = state.phase === "travel" && !state.ended;
      if (mesh.visible) mesh.position.set(...bezierPoint(edge.points, state.phaseProgress));
    });

    // The block breathes while it computes, then settles.
    const material = glow.current?.material as THREE.LineBasicMaterial | undefined;
    if (material) {
      const working =
        state.phase === "compute" ? 0.55 + 0.45 * Math.sin(state.phaseProgress * Math.PI * 4) : 0.5;
      material.opacity = state.ended
        ? 0
        : state.phase === "travel"
          ? 0.25 + 0.3 * state.phaseProgress
          : working;
    }
    // Its output appears in the last phase of the step.
    if (activePlane.current) activePlane.current.visible = state.phase === "emit" || state.ended;

    // Follow mode slides the camera along the flow, keeping the running block in view.
    const orbit = frame.controls as unknown as Controls | null;
    if (sim.follow && sim.playing && orbit) {
      const shift = (active.position[0] - orbit.target.x) * (1 - Math.exp(-delta * 3));
      frame.camera.position.x += shift;
      orbit.target.x += shift;
      orbit.update();
    }
  });

  if (!started || !plan) return null;
  return (
    <group>
      {done.map((item) => {
        const picture = viewOf(item)?.picture;
        return picture && item.glyph !== "merge" && item.glyph !== "ring" ? (
          <FeaturePlane key={item.id} item={item} picture={picture} />
        ) : null;
      })}
      <ActiveNeurons entries={columns} />

      {active && glowEdges ? (
        <lineSegments
          ref={glow}
          geometry={glowEdges}
          position={active.position}
          raycast={() => null}
        >
          <lineBasicMaterial color={palette.white} transparent opacity={0.5} toneMapped={false} />
        </lineSegments>
      ) : null}

      {active && activeView?.picture && active.glyph !== "merge" && active.glyph !== "ring" ? (
        <group ref={activePlane} visible={false}>
          <FeaturePlane item={active} picture={activeView.picture} />
        </group>
      ) : null}

      {incoming.map((edge, index) => (
        <mesh
          key={edge.id}
          ref={(mesh) => {
            pulses.current[index] = mesh;
          }}
          visible={false}
          raycast={() => null}
        >
          <sphereGeometry args={[0.07, 12, 10]} />
          <meshBasicMaterial color={activeColor} toneMapped={false} />
        </mesh>
      ))}
    </group>
  );
}
