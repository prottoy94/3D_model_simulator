"use client";

import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo, useRef } from "react";
import { formatCompact } from "@/core/format";
import { layoutModel2D, type GraphCard, type GraphFrame } from "@/core/layout";
import { FAMILY_COLOR_TOKEN } from "@/core/ops";
import { shapeText } from "@/core/shapes";
import { cn } from "@/lib/cn";
import { useForgeStore } from "@/stores/forge-store";
import { useModelStore } from "@/stores/model-store";

type CardNode = Node<{ card: GraphCard }, "card">;
type FrameNode = Node<{ frame: GraphFrame }, "frame">;

const statusMark: Record<GraphCard["status"], { glyph: string; className: string; label: string }> =
  {
    ok: { glyph: "✓", className: "text-ok", label: "computed and verified" },
    unverified: { glyph: "≈", className: "text-warn", label: "not verified" },
    mismatch: { glyph: "⚠", className: "text-error", label: "differs from the notebook's summary" },
    error: { glyph: "⚠", className: "text-error", label: "could not be computed" },
  };

function Card({ data }: NodeProps<CardNode>) {
  const { card } = data;
  const selected = useModelStore(
    (state) => state.selectedId === card.id || card.memberIds.includes(state.selectedId ?? ""),
  );
  const hovered = useModelStore((state) =>
    state.hoveredIds.some((id) => id === card.id || card.memberIds.includes(id)),
  );
  // Lens: cards outside the focus fade back.
  const faded = useModelStore(
    (state) =>
      state.focus !== null && !card.memberIds.some((id) => state.focus!.nodeIds.includes(id)),
  );
  const editable = useModelStore((state) => state.source === "forge");
  const handleClass = editable
    ? "!size-2.5 !border !border-accent !bg-deep"
    : "!size-1 !border-0 !bg-transparent";
  const color = `var(${FAMILY_COLOR_TOKEN[card.family]})`;
  const mark = statusMark[card.status];
  return (
    <div
      className={cn(
        "flex h-full w-full items-stretch overflow-hidden rounded-control border bg-glass-2 transition-[border-color,box-shadow] duration-150",
        selected ? "border-text-1" : hovered ? "border-accent" : "border-line",
        card.kind === "group" && "border-dashed",
        faded && "opacity-25",
      )}
      style={selected ? { boxShadow: `0 0 18px ${color}55` } : undefined}
    >
      <span
        aria-hidden
        className="w-1 shrink-0"
        style={{ background: color, boxShadow: `0 0 10px ${color}` }}
      />
      <div className="flex min-w-0 flex-1 flex-col justify-center px-2.5">
        <div className="flex items-baseline gap-1.5">
          <span className="truncate text-[12px] font-semibold text-text-1">{card.label}</span>
          {card.kind === "group" ? (
            <span className="shrink-0 text-[10px] text-text-3">{card.memberIds.length} layers</span>
          ) : null}
          <span className={cn("ml-auto shrink-0 text-[11px]", mark.className)} title={mark.label}>
            {mark.glyph}
          </span>
        </div>
        <div className="truncate font-mono text-[10px] text-text-3">
          {card.outShape ? shapeText(card.outShape) : "shape unknown"}
          {card.params > 0 ? ` · ${formatCompact(card.params)}` : ""}
        </div>
      </div>
      <Handle type="target" position={Position.Top} className={handleClass} />
      <Handle
        type="source"
        position={Position.Bottom}
        className="!size-1 !border-0 !bg-transparent"
      />
      {/* Skip connections leave and enter on the right so they curve around the cards between. */}
      <Handle
        id="skip-out"
        type="source"
        position={Position.Right}
        className="!size-1 !border-0 !bg-transparent"
      />
      <Handle
        id="skip-in"
        type="target"
        position={Position.Right}
        className="!size-1 !border-0 !bg-transparent"
      />
    </div>
  );
}

function Frame({ data }: NodeProps<FrameNode>) {
  const toggleGroup = useModelStore((state) => state.toggleGroup);
  return (
    <div className="h-full w-full rounded-control border border-line/80 bg-accent/[0.025]">
      <button
        type="button"
        title="Collapse this group"
        onClick={() => toggleGroup(data.frame.groupId)}
        className="hud-label nodrag nopan pointer-events-auto px-2 py-1 !text-[10px] hover:!text-accent"
      >
        {data.frame.label} ▾
      </button>
    </div>
  );
}

const nodeTypes = { card: Card, frame: Frame };

export default function GraphView() {
  const ir = useModelStore((state) => state.ir);
  const activeModelId = useModelStore((state) => state.activeModelId);
  const collapsed = useModelStore((state) => state.collapsed);
  const { select, hover, toggleGroup } = useModelStore.getState();
  // In Forge the graph is editable: drag between cards to connect, double-click a line to cut it.
  const editable = useModelStore((state) => state.source === "forge");
  const wrapper = useRef<HTMLDivElement>(null);

  const { nodes, edges } = useMemo(() => {
    if (!ir || !activeModelId) return { nodes: [], edges: [] };
    const layout = layoutModel2D(ir, activeModelId, new Set(collapsed));
    const frames: FrameNode[] = layout.frames.map((frame) => ({
      id: `frame:${frame.groupId}`,
      type: "frame",
      position: { x: frame.x, y: frame.y },
      width: frame.width,
      height: frame.height,
      data: { frame },
      draggable: false,
      selectable: false,
      zIndex: -1,
      // Only the label button reacts, so cards under the frame stay clickable.
      style: { pointerEvents: "none" },
    }));
    const cards: CardNode[] = layout.cards.map((card) => ({
      id: card.id,
      type: "card",
      position: { x: card.x, y: card.y },
      width: card.width,
      height: card.height,
      data: { card },
      draggable: false,
    }));
    const family = new Map(layout.cards.map((card) => [card.id, card.family]));
    const links: Edge[] = layout.edges.map((edge) => {
      const skip = edge.kind === "residual" || edge.kind === "skip" || edge.span > 1;
      return {
        id: edge.id,
        source: edge.from,
        target: edge.to,
        sourceHandle: skip ? "skip-out" : undefined,
        targetHandle: skip ? "skip-in" : undefined,
        style: {
          stroke: `var(${FAMILY_COLOR_TOKEN[family.get(edge.from) ?? "custom"]})`,
          strokeOpacity: skip ? 0.9 : 0.75,
          strokeWidth: 1.5,
          strokeDasharray: skip ? "5 4" : undefined,
        },
      };
    });
    return { nodes: [...frames, ...cards], edges: links };
  }, [ir, activeModelId, collapsed]);

  if (!ir || !activeModelId) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <p className="hud-label">No model to draw yet</p>
      </div>
    );
  }

  return (
    <div ref={wrapper} className="h-full w-full">
      <ReactFlow
        // A fresh instance per model so the view fits the new graph.
        key={activeModelId}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        colorMode="dark"
        fitView
        onInit={(instance) => {
          // A tall model does not fit at a readable zoom, so start at its first layers.
          const { zoom } = instance.getViewport();
          const height = wrapper.current?.clientHeight ?? 0;
          const bounds = instance.getNodesBounds(instance.getNodes());
          if (height > 0 && bounds.height * zoom > height) {
            void instance.setCenter(
              bounds.x + bounds.width / 2,
              bounds.y + height / zoom / 2 - 70 / zoom,
              { zoom },
            );
          }
        }}
        // Long models start at a readable zoom on their first layers instead of shrinking to fit.
        fitViewOptions={{ padding: 0.15, maxZoom: 1.1, minZoom: 0.6 }}
        minZoom={0.08}
        maxZoom={2}
        nodesConnectable={editable}
        onConnect={(link) => {
          if (link.source && link.target)
            useForgeStore.getState().connect(link.source, link.target);
        }}
        onEdgeDoubleClick={(_, edge) => {
          if (editable) useForgeStore.getState().disconnect(edge.source, edge.target);
        }}
        nodesDraggable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: false }}
        onNodeClick={(_, node) => {
          if (node.type === "card") select(node.id, "graph");
        }}
        onNodeDoubleClick={(_, node) => {
          if (node.type === "card" && (node as CardNode).data.card.kind === "group")
            toggleGroup(node.id);
        }}
        onNodeMouseEnter={(_, node) => {
          if (node.type === "card") hover([node.id]);
        }}
        onNodeMouseLeave={() => hover([])}
        onPaneClick={() => select(null, "graph")}
        style={{ background: "transparent" }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={24}
          size={1}
          color="var(--glass-border-strong)"
        />
        <Controls showInteractive={false} position="top-right" />
      </ReactFlow>
    </div>
  );
}
