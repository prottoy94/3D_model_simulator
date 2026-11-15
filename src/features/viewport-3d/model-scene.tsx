"use client";

import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, type RefObject } from "react";
import * as THREE from "three";
import { bezierPoint, type LayoutItem, type ModelLayout, type Vec3 } from "@/core/layout";
import { FAMILY_COLOR_TOKEN, type OpFamily } from "@/core/ops";
import { readToken } from "@/lib/tokens";
import { useModelStore } from "@/stores/model-store";
import { LabelProjector, type LabelRegistry } from "./labels";
import { SimLayer } from "./sim-layer";

const EDGE_SEGMENTS = 20;
const MAX_NEURONS = 18;

type Palette = {
  family: Record<OpFamily, THREE.Color>;
  warn: THREE.Color;
  error: THREE.Color;
  frame: THREE.Color;
};

function usePalette(): Palette {
  return useMemo(() => {
    const family = Object.fromEntries(
      Object.entries(FAMILY_COLOR_TOKEN).map(([name, token]) => [
        name,
        new THREE.Color(readToken(token as `--${string}`)),
      ]),
    ) as Record<OpFamily, THREE.Color>;
    return {
      family,
      warn: new THREE.Color(readToken("--warn")),
      error: new THREE.Color(readToken("--error")),
      frame: new THREE.Color(readToken("--op-shape")),
    };
  }, []);
}

/** The outline shows trust: rose for a mismatch or error, amber for unverified. */
function outlineColor(item: LayoutItem, palette: Palette): THREE.Color {
  if (item.status === "error" || item.status === "mismatch") return palette.error;
  if (item.status === "unverified") return palette.warn;
  return palette.family[item.family];
}

/** The 12 edges of a box as line-segment vertices. */
function boxEdges(min: Vec3, max: Vec3, out: number[]): void {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  const corners: Vec3[] = [
    [x0, y0, z0],
    [x1, y0, z0],
    [x1, y0, z1],
    [x0, y0, z1],
    [x0, y1, z0],
    [x1, y1, z0],
    [x1, y1, z1],
    [x0, y1, z1],
  ];
  const pairs = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7];
  for (const index of pairs) out.push(...corners[index]!);
}

const itemMin = (item: LayoutItem, grow = 0): Vec3 =>
  item.position.map((value, axis) => value - item.size[axis]! / 2 - grow) as Vec3;
const itemMax = (item: LayoutItem, grow = 0): Vec3 =>
  item.position.map((value, axis) => value + item.size[axis]! / 2 + grow) as Vec3;

function lineGeometry(positions: number[], colors?: number[]): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  if (colors) geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return geometry;
}

/** Frees a geometry's GPU buffers when it is replaced or the scene unmounts. */
function useDispose(value: { dispose(): void }): void {
  useEffect(() => () => value.dispose(), [value]);
}

type InstancedProps = {
  items: LayoutItem[];
  palette: Palette;
  children: React.ReactNode;
  opacity: number;
  /** Per-item transform: returns scale; position is the item's centre. */
  scale: (item: LayoutItem) => Vec3;
  /** Brightness per item: 1 in the Lens focus, lower for its other copies, lowest outside. */
  dim: (item: LayoutItem) => number;
  rotation?: [number, number, number];
};

/** One draw call per glyph kind. Clicks and hovers resolve through the instance id. */
function Glyphs({ items, palette, children, opacity, scale, rotation, dim }: InstancedProps) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const select = useModelStore((state) => state.select);
  const hover = useModelStore((state) => state.hover);
  const toggleGroup = useModelStore((state) => state.toggleGroup);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const tint = new THREE.Color();
    const dummy = new THREE.Object3D();
    items.forEach((item, index) => {
      dummy.position.set(...item.position);
      dummy.scale.set(...scale(item));
      dummy.rotation.set(...(rotation ?? [0, 0, 0]));
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
      mesh.setColorAt(index, tint.copy(palette.family[item.family]).multiplyScalar(dim(item)));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [items, palette, scale, rotation, dim]);

  if (items.length === 0) return null;
  const at = (event: ThreeEvent<MouseEvent>) =>
    event.instanceId === undefined ? undefined : items[event.instanceId];

  return (
    <instancedMesh
      // A new mesh when the count changes keeps the instance buffers the right size.
      key={items.length}
      ref={ref}
      args={[undefined, undefined, items.length]}
      onClick={(event) => {
        event.stopPropagation();
        const item = at(event);
        if (item) select(item.id, "viewport");
      }}
      onDoubleClick={(event) => {
        event.stopPropagation();
        const item = at(event);
        if (item?.kind === "group") toggleGroup(item.id);
      }}
      onPointerMove={(event) => {
        event.stopPropagation();
        const item = at(event);
        if (item) hover([item.id]);
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        hover([]);
        document.body.style.cursor = "";
      }}
    >
      {children}
      <meshBasicMaterial transparent opacity={opacity} depthWrite={false} toneMapped={false} />
    </instancedMesh>
  );
}

const boxScale = (item: LayoutItem): Vec3 => item.size;
const mergeScale = (item: LayoutItem): Vec3 => [
  item.size[0] * 0.7,
  item.size[1] * 0.7,
  item.size[2] * 0.7,
];
const ringScale = (item: LayoutItem): Vec3 => [
  item.size[1] / 2,
  item.size[1] / 2,
  item.size[1] / 2,
];
const RING_ROTATION: [number, number, number] = [0, Math.PI / 2, 0];

function Neurons({ items, palette }: { items: LayoutItem[]; palette: Palette }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const dots = useMemo(() => {
    const result: Array<{ position: Vec3; color: THREE.Color; radius: number }> = [];
    for (const item of items) {
      const count = Math.max(1, Math.min(MAX_NEURONS, item.units ?? 1));
      const step = item.size[1] / count;
      const radius = Math.min(0.09, step * 0.36);
      for (let i = 0; i < count; i++) {
        result.push({
          position: [
            item.position[0],
            item.position[1] - item.size[1] / 2 + step * (i + 0.5),
            item.position[2],
          ],
          color: palette.family[item.family],
          radius,
        });
      }
    }
    return result;
  }, [items, palette]);

  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    dots.forEach((dot, index) => {
      dummy.position.set(...dot.position);
      dummy.scale.setScalar(dot.radius);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
      mesh.setColorAt(index, dot.color);
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

function Highlight({ item, color, grow }: { item: LayoutItem; color: string; grow: number }) {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    boxEdges(itemMin(item, grow), itemMax(item, grow), positions);
    return lineGeometry(positions);
  }, [item, grow]);
  useDispose(geometry);
  return (
    <lineSegments geometry={geometry} raycast={() => null}>
      <lineBasicMaterial color={color} toneMapped={false} />
    </lineSegments>
  );
}

type Controls = {
  target: THREE.Vector3;
  update(): void;
  addEventListener(type: "start", listener: () => void): void;
  removeEventListener(type: "start", listener: () => void): void;
};

/**
 * Frames the whole model: when it changes, when Overview is pressed, and when
 * the viewport is resized while the user has not moved the camera themselves.
 */
function CameraRig({ bounds }: { bounds: ModelLayout["bounds"] }) {
  const fitRequest = useModelStore((state) => state.fitRequest);
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls) as unknown as Controls | null;
  const width = useThree((state) => state.size.width);
  const height = useThree((state) => state.size.height);
  const goal = useRef<{ position: THREE.Vector3; target: THREE.Vector3 } | null>(null);
  /** True once the user has dragged or zoomed since the last fit. */
  const touched = useRef(false);
  const lastRequest = useRef(-1);
  const boundsRef = useRef(bounds);
  useEffect(() => {
    boundsRef.current = bounds;
  }, [bounds]);

  // The user's own gesture always wins over a camera move in progress.
  useEffect(() => {
    if (!controls) return;
    const onStart = () => {
      touched.current = true;
      goal.current = null;
    };
    controls.addEventListener("start", onStart);
    return () => controls.removeEventListener("start", onStart);
  }, [controls]);

  useEffect(() => {
    const requested = lastRequest.current !== fitRequest;
    lastRequest.current = fitRequest;
    // A resize alone must not undo a view the user has set up.
    if (!requested && touched.current) return;
    if (width === 0 || height === 0) return;
    touched.current = false;

    const { min, max } = boundsRef.current;
    const centre = new THREE.Vector3(
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    );
    // From the input side, slightly above: the faces that carry each layer's output face the camera.
    const direction = new THREE.Vector3(-1, 0.42, 0.95).normalize();
    // Find the closest distance at which all eight corners of the model are inside the view.
    const probe = (camera as THREE.PerspectiveCamera).clone();
    probe.aspect = width / height;
    probe.updateProjectionMatrix();
    const corner = new THREE.Vector3();
    const fits = (distance: number): boolean => {
      probe.position.copy(centre).addScaledVector(direction, distance);
      probe.lookAt(centre);
      probe.updateMatrixWorld();
      for (let i = 0; i < 8; i++) {
        corner
          .set(i & 1 ? max[0] : min[0], i & 2 ? max[1] : min[1], i & 4 ? max[2] : min[2])
          .project(probe);
        // Leave room for the toolbar at the bottom and the banner at the top.
        if (Math.abs(corner.x) > 0.9 || corner.y > 0.72 || corner.y < -0.7 || corner.z > 1)
          return false;
      }
      return true;
    };
    let near = 3;
    let far = 400;
    for (let i = 0; i < 24; i++) {
      const middle = (near + far) / 2;
      if (fits(middle)) far = middle;
      else near = middle;
    }
    goal.current = { position: centre.clone().addScaledVector(direction, far), target: centre };
  }, [fitRequest, camera, width, height]);

  useFrame((_, delta) => {
    const move = goal.current;
    if (!move || !controls) return;
    const ease = 1 - Math.exp(-Math.min(delta, 0.1) * 6);
    camera.position.lerp(move.position, ease);
    controls.target.lerp(move.target, ease);
    controls.update();
    if (camera.position.distanceTo(move.position) < 0.02) goal.current = null;
  });
  return null;
}

export function ModelScene({
  layout,
  labels,
}: {
  layout: ModelLayout;
  labels: RefObject<LabelRegistry>;
}) {
  const palette = usePalette();
  const selectedId = useModelStore((state) => state.selectedId);
  const hoveredIds = useModelStore((state) => state.hoveredIds);
  const select = useModelStore((state) => state.select);

  const focus = useModelStore((state) => state.focus);
  // Lens: which blocks are in the focus, which are other copies of it, and the box around the focus.
  const lens = useMemo(() => {
    if (!focus) return null;
    const inFocus = new Set(focus.nodeIds);
    const echo = new Set(focus.allNodeIds);
    const main = new Set<string>();
    const copies = new Set<string>();
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (const item of layout.items) {
      if (item.memberIds.some((id) => inFocus.has(id))) {
        main.add(item.id);
        for (let axis = 0; axis < 3; axis++) {
          min[axis] = Math.min(min[axis]!, item.position[axis]! - item.size[axis]! / 2 - 0.3);
          max[axis] = Math.max(max[axis]!, item.position[axis]! + item.size[axis]! / 2 + 0.3);
        }
      } else if (item.memberIds.some((id) => echo.has(id))) copies.add(item.id);
    }
    return main.size > 0 ? { main, copies, bounds: { min, max } } : null;
  }, [focus, layout]);
  const dim = useCallback(
    (item: LayoutItem) =>
      !lens ? 1 : lens.main.has(item.id) ? 1 : lens.copies.has(item.id) ? 0.42 : 0.1,
    [lens],
  );
  const focusFrame = useMemo(() => {
    if (!lens) return null;
    const positions: number[] = [];
    boxEdges(lens.bounds.min, lens.bounds.max, positions);
    return lineGeometry(positions);
  }, [lens]);
  useEffect(() => () => focusFrame?.dispose(), [focusFrame]);

  const { boxes, merges, rings, columns } = useMemo(
    () => ({
      boxes: layout.items.filter((item) => item.glyph !== "merge" && item.glyph !== "ring"),
      merges: layout.items.filter((item) => item.glyph === "merge"),
      rings: layout.items.filter((item) => item.glyph === "ring"),
      columns: layout.items.filter((item) => item.glyph === "neuron-column" && item.units !== null),
    }),
    [layout],
  );

  const outlines = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    for (const item of layout.items) {
      if (item.glyph === "merge" || item.glyph === "ring") continue;
      const before = positions.length;
      boxEdges(itemMin(item), itemMax(item), positions);
      const color = outlineColor(item, palette);
      const level = dim(item);
      for (let i = before; i < positions.length; i += 3) {
        colors.push(color.r * level, color.g * level, color.b * level);
      }
    }
    return lineGeometry(positions, colors);
  }, [layout, palette, dim]);

  const edges = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    const byId = new Map(layout.items.map((item) => [item.id, item]));
    for (const edge of layout.edges) {
      const source = byId.get(edge.from);
      const color = source ? palette.family[source.family] : palette.frame;
      const target = byId.get(edge.to);
      const strength =
        (edge.arc ? 0.75 : 0.45) * Math.min(source ? dim(source) : 1, target ? dim(target) : 1);
      let previous = bezierPoint(edge.points, 0);
      for (let i = 1; i <= EDGE_SEGMENTS; i++) {
        const next = bezierPoint(edge.points, i / EDGE_SEGMENTS);
        positions.push(...previous, ...next);
        for (let k = 0; k < 2; k++)
          colors.push(color.r * strength, color.g * strength, color.b * strength);
        previous = next;
      }
    }
    return lineGeometry(positions, colors);
  }, [layout, palette, dim]);

  const frames = useMemo(() => {
    const positions: number[] = [];
    for (const frame of layout.frames) boxEdges(frame.min, frame.max, positions);
    return lineGeometry(positions);
  }, [layout]);
  useDispose(outlines);
  useDispose(edges);
  useDispose(frames);

  const visibleOf = useMemo(() => {
    const map = new Map<string, LayoutItem>();
    for (const item of layout.items) {
      map.set(item.id, item);
      for (const member of item.memberIds) if (!map.has(member)) map.set(member, item);
    }
    return map;
  }, [layout]);

  const selected = selectedId ? visibleOf.get(selectedId) : undefined;
  const hovered = [
    ...new Set(
      hoveredIds.map((id) => visibleOf.get(id)).filter((item): item is LayoutItem => !!item),
    ),
  ];

  return (
    <group onPointerMissed={() => select(null, "viewport")}>
      <CameraRig bounds={lens?.bounds ?? layout.bounds} />
      {focusFrame ? (
        <lineSegments geometry={focusFrame} raycast={() => null}>
          <lineBasicMaterial color={readToken("--accent")} toneMapped={false} />
        </lineSegments>
      ) : null}

      <Glyphs items={boxes} palette={palette} opacity={0.14} scale={boxScale} dim={dim}>
        <boxGeometry args={[1, 1, 1]} />
      </Glyphs>
      <Glyphs items={merges} palette={palette} opacity={0.85} scale={mergeScale} dim={dim}>
        <octahedronGeometry args={[1, 0]} />
      </Glyphs>
      <Glyphs
        items={rings}
        dim={dim}
        palette={palette}
        opacity={0.8}
        scale={ringScale}
        rotation={RING_ROTATION}
      >
        <torusGeometry args={[1, 0.07, 10, 48]} />
      </Glyphs>
      <Neurons items={columns} palette={palette} />

      <lineSegments geometry={outlines} raycast={() => null}>
        <lineBasicMaterial vertexColors toneMapped={false} />
      </lineSegments>
      <lineSegments geometry={edges} raycast={() => null}>
        <lineBasicMaterial vertexColors toneMapped={false} transparent opacity={0.9} />
      </lineSegments>
      <lineSegments geometry={frames} raycast={() => null}>
        <lineBasicMaterial color={palette.frame} transparent opacity={0.28} />
      </lineSegments>

      {hovered
        .filter((item) => item.id !== selected?.id)
        .map((item) => (
          <Highlight key={item.id} item={item} color={readToken("--accent")} grow={0.03} />
        ))}
      {selected ? <Highlight item={selected} color={readToken("--text-1")} grow={0.05} /> : null}

      <SimLayer layout={layout} />
      <LabelProjector layout={layout} registry={labels} />
    </group>
  );
}
