"use client";

import { useFrame, useThree } from "@react-three/fiber";
import { useMemo, type RefObject } from "react";
import * as THREE from "three";
import { formatCompact } from "@/core/format";
import type { LayoutItem, ModelLayout, Vec3 } from "@/core/layout";
import { shapeText } from "@/core/shapes";
import { useModelStore } from "@/stores/model-store";
import { useSimStore } from "@/stores/sim-store";

/** DOM elements of the labels, keyed like the anchors below. Shared by the two halves. */
export type LabelRegistry = Map<string, HTMLElement>;

/** Above this many glyphs, layer labels only show for the hovered and selected ones. */
const LABEL_LIMIT = 90;

const itemKey = (id: string) => `item:${id}`;
const frameKey = (id: string) => `frame:${id}`;

function useVisibleItems(layout: ModelLayout) {
  const selectedId = useModelStore((state) => state.selectedId);
  const hoveredIds = useModelStore((state) => state.hoveredIds);
  const showLabels = useModelStore((state) => state.showLabels);
  // The block that is running in the simulation gets a full label as well.
  const runningId = useSimStore((state) =>
    state.status === "idle" ? null : (state.plan?.steps[state.step]?.itemId ?? null),
  );
  const focus = useModelStore((state) => state.focus);
  return useMemo(() => {
    const focusIds = focus ? new Set(focus.nodeIds) : null;
    const strong = new Set<string>();
    for (const item of layout.items) {
      const ids = [item.id, ...item.memberIds];
      if (item.id === runningId || ids.some((id) => id === selectedId || hoveredIds.includes(id))) {
        strong.add(item.id);
      }
    }
    const all = showLabels && layout.items.length <= LABEL_LIMIT;
    const inFocus = focusIds
      ? (item: LayoutItem) => item.memberIds.some((id) => focusIds.has(id))
      : () => true;
    return {
      strong,
      shown: layout.items.filter((item) => strong.has(item.id) || (all && inFocus(item))),
    };
  }, [layout, selectedId, hoveredIds, showLabels, runningId, focus]);
}

/**
 * The labels themselves: plain DOM over the canvas. They are positioned every
 * frame by <LabelProjector> through refs, so React never re-renders per frame.
 */
export function LabelOverlay({
  layout,
  registry,
}: {
  layout: ModelLayout;
  registry: RefObject<LabelRegistry>;
}) {
  const { strong, shown } = useVisibleItems(layout);
  const toggleGroup = useModelStore((state) => state.toggleGroup);
  const register = (key: string) => (element: HTMLElement | null) => {
    if (element) registry.current.set(key, element);
    else registry.current.delete(key);
  };

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      {layout.frames.map((frame) => (
        <button
          key={frame.groupId}
          ref={register(frameKey(frame.groupId))}
          type="button"
          title="Collapse this group"
          data-snapshot-label
          onClick={() => toggleGroup(frame.groupId)}
          style={{ opacity: 0 }}
          className="hud-label pointer-events-auto absolute left-0 top-0 origin-bottom-left whitespace-nowrap rounded-t-[6px] border border-b-0 border-line bg-glass-1 px-2 py-0.5 !text-[10px] will-change-transform hover:!text-accent"
        >
          {frame.label} <span aria-hidden>▾</span>
        </button>
      ))}
      {shown.map((item) => {
        const isStrong = strong.has(item.id);
        return (
          <div
            key={item.id}
            ref={register(itemKey(item.id))}
            data-strong={isStrong}
            data-snapshot-label
            style={{ opacity: 0 }}
            className={
              isStrong
                ? "absolute left-0 top-0 z-10 whitespace-nowrap rounded-control border border-line-strong bg-glass-2 px-2 py-1 text-center will-change-transform"
                : "absolute left-0 top-0 whitespace-nowrap text-center will-change-transform"
            }
          >
            <div
              className={isStrong ? "text-xs font-semibold text-text-1" : "text-[11px] text-text-2"}
            >
              {item.label}
            </div>
            {item.outShape ? (
              <div className="font-mono text-[10px] text-text-3">
                {shapeText(item.outShape)}
                {isStrong && item.params > 0 ? ` · ${formatCompact(item.params)} params` : ""}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

type Anchor = { key: string; position: Vec3; kind: "item" | "frame"; strong: boolean };

const topOf = (item: LayoutItem): Vec3 => [
  item.position[0],
  item.position[1] + item.size[1] / 2 + 0.12,
  item.position[2],
];

/** Lives inside the canvas: projects each label's anchor to the screen and hides the ones that would collide. */
export function LabelProjector({
  layout,
  registry,
}: {
  layout: ModelLayout;
  registry: RefObject<LabelRegistry>;
}) {
  const { strong, shown } = useVisibleItems(layout);
  const size = useThree((state) => state.size);
  const anchors = useMemo<Anchor[]>(
    () => [
      ...shown.map((item) => ({
        key: itemKey(item.id),
        position: topOf(item),
        kind: "item" as const,
        strong: strong.has(item.id),
      })),
      ...layout.frames.map((frame) => ({
        key: frameKey(frame.groupId),
        position: [frame.min[0], frame.max[1], frame.max[2]] as Vec3,
        kind: "frame" as const,
        strong: false,
      })),
    ],
    [layout, shown, strong],
  );
  const scratch = useMemo(() => new THREE.Vector3(), []);

  useFrame(({ camera }) => {
    const placed: Array<{ x: number; y: number; w: number; h: number }> = [];
    const entries = anchors
      .map((anchor) => {
        scratch.set(...anchor.position);
        const distance = scratch.distanceTo(camera.position);
        scratch.project(camera);
        return {
          anchor,
          distance,
          x: (scratch.x * 0.5 + 0.5) * size.width,
          y: (-scratch.y * 0.5 + 0.5) * size.height,
          inFront: scratch.z < 1,
        };
      })
      // Selected and hovered labels claim their space first, then the nearest ones.
      .sort((a, b) => Number(b.anchor.strong) - Number(a.anchor.strong) || a.distance - b.distance);

    for (const { anchor, distance, x, y, inFront } of entries) {
      const element = registry.current.get(anchor.key);
      if (!element) continue;
      const scale = anchor.strong ? 1 : Math.max(0.55, Math.min(1, 16 / distance));
      const w = element.offsetWidth * scale;
      const h = element.offsetHeight * scale;
      const onScreen = inFront && x > -w && x < size.width + w && y > 0 && y < size.height + h;
      // Frame tabs hang from their corner; layer labels are centred above the glyph.
      const left = anchor.kind === "frame" ? x : x - w / 2;
      const box = { x: left, y: y - h, w, h };
      const collides =
        !anchor.strong &&
        placed.some(
          (other) =>
            box.x < other.x + other.w + 4 &&
            other.x < box.x + box.w + 4 &&
            box.y < other.y + other.h + 2 &&
            other.y < box.y + box.h + 2,
        );
      if (!onScreen || collides || (!anchor.strong && distance > 70)) {
        element.style.opacity = "0";
        element.style.pointerEvents = "none";
        continue;
      }
      placed.push(box);
      element.style.opacity = anchor.strong ? "1" : "0.85";
      element.style.pointerEvents = anchor.kind === "frame" ? "auto" : "none";
      element.style.transformOrigin = "0 0";
      element.style.transform = `translate(${left.toFixed(1)}px, ${(y - h).toFixed(1)}px) scale(${scale.toFixed(3)})`;
    }
  });
  return null;
}
