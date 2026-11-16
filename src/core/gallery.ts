/**
 * Models offered in the gallery: the hand-written Wire IR examples, plus
 * models written in the Draft format.
 */
import { draftToWire, parseDraft, type Draft } from "./ai/draft";
import { WireBuilder } from "./ir/builder";
import { EXAMPLE_IRS } from "./ir/examples";
import type { WireIR } from "./ir/wire";

/**
 * FireNet, written out by hand from the reference notebook N6 (cell 9). It is
 * the golden check for the attention, group-norm, resize, anchor and
 * shared-weight features: the notebook prints 32.18 M parameters and 9,261 anchors.
 */
export function fireNetDraft(): Draft {
  type Node = Record<string, unknown>;
  const nodes: Node[] = [];
  let group: string | null = null;
  const add = (id: string, op: string, inputs: string[], params: Node = {}, extra: Node = {}) => {
    nodes.push({ id, op, inputs, params, group, lines: [9, 1, 1], ...extra });
    return id;
  };
  const zoo = "torchvision.models.convnext_tiny";
  add("in", "input", [], { shape: [3, 224, 224] });
  add("base", "backbone", ["in"], { zoo, inputSize: [224, 224] });
  add("f2", "backbone", ["base"], { zoo, inputSize: [224, 224], stage: 2 });
  add("f3", "backbone", ["base"], { zoo, inputSize: [224, 224], stage: 3 });

  // MultiScaleAttention: a 1×1 adapter and cascaded attention on each of the three scales.
  group = "attention";
  const levels = [
    ["f2", 3],
    ["f3", 4],
    ["base", 5],
  ] as const;
  for (const [source, level] of levels) {
    add(`adapt${level}`, "conv2d", [source], { filters: 256, kernel: 1 });
    add(`ca${level}`, "channel_attention", [`adapt${level}`], { reduction: 16, minHidden: 8 });
    add(`p${level}`, "spatial_attention", [`ca${level}`], { kernel: 7 });
  }
  add("up4", "upsample", ["p4"], { size: [28, 28], mode: "bilinear" });
  add("up5", "upsample", ["p5"], { size: [28, 28], mode: "bilinear" });
  add("cat", "concat", ["p3", "up4", "up5"], { axis: 1 });
  add("fuse_conv", "conv2d", ["cat"], { filters: 256, kernel: 1, bias: false });
  add("fuse_gn", "groupnorm", ["fuse_conv"], { groups: 32 });
  add("fused", "relu", ["fuse_gn"]);

  // ClassificationHead: average and maximum of the fused map, then two dense layers.
  group = "cls_head";
  add("avg", "global_avg_pool", ["fused"], { keepDims: false });
  add("max", "global_max_pool", ["fused"], { keepDims: false });
  add("both", "concat", ["avg", "max"], { axis: 1 });
  add("fc1", "dense", ["both"], { units: 256, activation: "gelu" });
  add("cls_logits", "dense", ["fc1"], { units: 2 });

  // DetectionHead: two towers of three conv + GroupNorm + ReLU, shared by the three scales.
  group = "det_head";
  const clsParts: string[] = [];
  const boxParts: string[] = [];
  const anchorParts: string[] = [];
  for (const [, level] of levels) {
    const shared = (id: string) => (level === 3 ? {} : { sharedWith: `${id}3` });
    const cells = { 3: 28 * 28, 4: 14 * 14, 5: 7 * 7 }[level];
    for (const tower of ["cls", "box"]) {
      let previous = `p${level}`;
      for (let i = 1; i <= 3; i++) {
        add(
          `${tower}_conv${i}_${level}`,
          "conv2d",
          [previous],
          { filters: 256, kernel: 3, padding: 1, bias: false },
          shared(`${tower}_conv${i}_`),
        );
        add(
          `${tower}_gn${i}_${level}`,
          "groupnorm",
          [`${tower}_conv${i}_${level}`],
          { groups: 32 },
          shared(`${tower}_gn${i}_`),
        );
        previous = add(`${tower}_relu${i}_${level}`, "relu", [`${tower}_gn${i}_${level}`]);
      }
      const outputs = tower === "cls" ? 2 : 4;
      add(
        `${tower}_out_${level}`,
        "conv2d",
        [previous],
        { filters: 9 * outputs, kernel: 3, padding: 1 },
        shared(`${tower}_out_`),
      );
      (tower === "cls" ? clsParts : boxParts).push(
        add(`${tower}_flat_${level}`, "reshape", [`${tower}_out_${level}`], {
          shape: [cells * 9, outputs],
        }),
      );
    }
    anchorParts.push(
      add(`anchors${level}`, "anchor_generator", [`p${level}`], {
        perLocation: 9,
        stride: 2 ** level,
      }),
    );
  }
  add("det_cls", "concat", clsParts, { axis: 1 });
  add("det_box", "concat", boxParts, { axis: 1 });
  add("anchors", "concat", anchorParts, { axis: 1 });

  const parsed = parseDraft(
    JSON.stringify({
      title: "FireNet",
      summary:
        "ConvNeXt-Tiny features at three scales pass through channel and spatial attention, then feed a classification head and a RetinaNet-style detection head whose towers are shared across the scales.",
      framework: "pytorch",
      task: "multi-task",
      models: [
        {
          id: "firenet",
          name: "FireNet",
          groups: [
            { id: "attention", label: "Multi-scale attention", kind: "module" },
            { id: "cls_head", label: "Classification head", kind: "head" },
            { id: "det_head", label: "Detection head", kind: "head" },
          ],
          nodes,
          outputs: ["cls_logits", "det_cls", "det_box"],
        },
      ],
    }),
  );
  if (!parsed.ok) throw new Error(parsed.errors.join("\n"));
  return parsed.value;
}

/**
 * A small one-stage detector with everything that comes after the heads: anchors,
 * box decoding, non-maximum suppression and RoI align. It is not a published
 * model; it exists to show those steps on something small enough to compute.
 */
export function tinyDetectorIR(): WireIR {
  const b = new WireBuilder({
    title: "Tiny detector",
    summary:
      "A small one-stage object detector, made for this gallery. Three convolutions give a 16×16 feature map. Two heads predict, for each of 768 anchor boxes, a score and four offsets. The offsets are applied to the anchors, overlapping boxes are removed, and the features under each remaining box are cut out.",
    framework: "keras",
  })
    .model("m_detector", "Tiny detector", "Backbone, two heads and the steps after them")
    .node("n_in", "input", { shape: [128, 128, 3] }, { label: "Input 128×128×3", trainable: "n/a" })
    .group("g_backbone", "Backbone", { kind: "block" });

  let previous = "n_in";
  [16, 32, 64].forEach((filters, index) => {
    const id = `n_conv${index + 1}`;
    b.node(
      id,
      "conv2d",
      { filters, kernel: 3, stride: 2, padding: "same", activation: "relu" },
      { from: [previous], label: `Conv ${index + 1} · ${filters}@3×3 /2` },
    );
    previous = id;
  });

  b.group("g_heads", "Heads", { kind: "block" })
    .node(
      "n_box_head",
      "conv2d",
      { filters: 12, kernel: 1 },
      {
        from: [previous],
        label: "Box head · 3 anchors × 4 offsets",
      },
    )
    .node(
      "n_deltas",
      "reshape",
      { shape: [768, 4] },
      { from: ["n_box_head"], label: "Offsets 768×4" },
    )
    .node(
      "n_cls_head",
      "conv2d",
      { filters: 6, kernel: 1 },
      {
        from: [previous],
        label: "Class head · 3 anchors × 2 classes",
      },
    )
    .node(
      "n_logits",
      "reshape",
      { shape: [768, 2] },
      { from: ["n_cls_head"], label: "Logits 768×2" },
    )
    .node("n_scores", "softmax", {}, { from: ["n_logits"], label: "Scores" })
    .group("g_post", "After the heads", { kind: "block" })
    .node(
      "n_anchors",
      "anchor_generator",
      { perLocation: 3, stride: 8 },
      {
        from: [previous],
        label: "Anchors · 768 boxes",
        trainable: "n/a",
      },
    )
    .node(
      "n_boxes",
      "box_decode",
      {},
      {
        from: ["n_deltas", "n_anchors"],
        label: "Decode boxes",
        trainable: "n/a",
      },
    )
    .node(
      "n_kept",
      "nms",
      { iouThreshold: 0.5, scoreThreshold: 0.05, maxOutput: 20 },
      {
        from: ["n_boxes", "n_scores"],
        label: "NMS · keep K boxes",
        trainable: "n/a",
      },
    )
    .node(
      "n_crops",
      "roi_align",
      { pooled: 7 },
      {
        from: [previous, "n_kept"],
        label: "RoI align 7×7",
        trainable: "n/a",
      },
    )
    .outputs("n_kept", "n_crops");
  return b.build();
}

export const GALLERY: Record<string, () => WireIR> = {
  ...EXAMPLE_IRS,
  firenet: () => draftToWire(fireNetDraft()),
  "tiny-detector": tinyDetectorIR,
};
