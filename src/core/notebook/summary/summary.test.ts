import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { extractOutputFacts, parseSummaries } from "./index";
import { parseShape, splitNameAndType } from "./shared";

/** Real outputs captured from the reference notebooks (docs/07-NOTEBOOK-PATTERNS.md §4). */
function fixture(name: string): string {
  return readFileSync(path.join(process.cwd(), "fixtures", "summaries", `${name}.txt`), "utf8");
}

function only(name: string) {
  const summaries = parseSummaries(fixture(name));
  expect(summaries).toHaveLength(1);
  return summaries[0];
}

describe("Keras 3 summaries from the reference notebooks", () => {
  it("N1 AlexNet: rows, shapes and totals", () => {
    const summary = only("n1-alexnet-vgg-cell16");
    expect(summary.format).toBe("keras3");
    expect(summary.modelName).toBe("AlexNet");
    expect(summary.totals).toEqual({ total: 46781192, trainable: 46780488, nonTrainable: 704 });

    const [conv1, bn1, pool1, conv2] = summary.rows;
    expect(conv1).toMatchObject({
      name: "conv2d",
      type: "Conv2D",
      outputShape: [null, 54, 54, 96],
      params: 34944,
    });
    // The name wraps onto a second physical line in the table.
    expect(bn1).toMatchObject({
      name: "batch_normalization",
      type: "BatchNormalization",
      params: 384,
    });
    expect(pool1.outputShape).toEqual([null, 26, 26, 96]);
    expect(conv2).toMatchObject({ outputShape: [null, 26, 26, 256], params: 614656 });

    expect(summary.rows).toHaveLength(16);
    expect(summary.rows.reduce((sum, row) => sum + (row.params ?? 0), 0)).toBe(46781192);
  });

  it("N1 VGG16 variant", () => {
    const summary = only("n1-alexnet-vgg-cell28");
    expect(summary.modelName).toBe("VGG16");
    expect(summary.totals).toEqual({ total: 14981448, trainable: 14981448, nonTrainable: 0 });
    expect(summary.rows.filter((row) => row.type === "Conv2D")).toHaveLength(13);
    expect(summary.rows.reduce((sum, row) => sum + (row.params ?? 0), 0)).toBe(14981448);
  });

  it("N2 EfficientNetB0 in both fine-tuning phases", () => {
    const phase1 = only("n2-efficientnet-cell19");
    const phase2 = only("n2-efficientnet-cell23");
    expect(phase1.totals).toEqual({ total: 4214571, trainable: 165000, nonTrainable: 4049571 });
    expect(phase2.totals).toEqual({ total: 4214571, trainable: 1661160, nonTrainable: 2553411 });
    expect(phase1.rows.length).toBeGreaterThan(200);
    expect(phase1.rows.reduce((sum, row) => sum + (row.params ?? 0), 0)).toBe(4214571);
  });

  it("N3 MobileNetV2", () => {
    const summary = only("n3-mobilenetv2-cell18");
    expect(summary.totals).toEqual({ total: 2422984, trainable: 165000, nonTrainable: 2257984 });
  });

  it("N4 mini-ViT: truncated names, dynamic dims and connections", () => {
    const summary = only("n4-vit-cell18");
    expect(summary.modelName).toBe("ViT_scratch");
    expect(summary.totals.total).toBe(404040);

    const patches = summary.rows.find((row) => row.type === "Patches");
    expect(patches?.outputShape).toEqual([null, null, 768]);
    expect(patches?.connectedTo).toEqual(["input_layer"]);

    const encoder = summary.rows.find((row) => row.type === "PatchEncoder");
    expect(encoder).toMatchObject({ outputShape: [null, 196, 64], params: 61760 });

    // "multi_head_attenti… (MultiHeadAttentio…" is cut short in the table.
    const attention = summary.rows.filter((row) => row.type.startsWith("MultiHeadAttentio"));
    expect(attention).toHaveLength(4);
    expect(attention[0]).toMatchObject({ truncated: true, params: 66368 });
    expect(attention[0].connectedTo).toHaveLength(2);

    const input = summary.rows[0];
    expect(input).toMatchObject({ type: "InputLayer", outputShape: [null, 224, 224, 3] });
  });

  it("N5 R-CNN family", () => {
    expect(only("n5-rcnn-cell26")).toMatchObject({
      modelName: "RCNN_Classifier",
      totals: { total: 1143113 },
    });
    expect(only("n5-rcnn-cell43")).toMatchObject({
      modelName: "FastRCNN",
      totals: { total: 897869 },
    });
    expect(only("n5-rcnn-cell58")).toMatchObject({ modelName: "RPN", totals: { total: 246637 } });
  });
});

describe("Keras 2 summaries", () => {
  const text = [
    'Model: "sequential"',
    "_________________________________________________________________",
    " Layer (type)                Output Shape              Param #   ",
    "=================================================================",
    " conv2d (Conv2D)             (None, 54, 54, 96)        34944     ",
    "                                                                 ",
    " a_very_long_layer_name_that_wraps  (None, 26, 26, 96)  0        ",
    " (MaxPooling2D)                                                  ",
    "                                                                 ",
    " dense (Dense)               (None, 8)                 32776     ",
    "                                                                 ",
    "=================================================================",
    "Total params: 67,720",
    "Trainable params: 67,720",
    "Non-trainable params: 0",
    "_________________________________________________________________",
  ].join("\n");

  it("parses rows and totals", () => {
    const summaries = parseSummaries(text);
    expect(summaries).toHaveLength(1);
    const summary = summaries[0];
    expect(summary).toMatchObject({ format: "keras2", modelName: "sequential" });
    expect(summary.totals).toEqual({ total: 67720, trainable: 67720, nonTrainable: 0 });
    expect(summary.rows.map((row) => row.type)).toEqual(["Conv2D", "MaxPooling2D", "Dense"]);
    expect(summary.rows[0]).toMatchObject({ outputShape: [null, 54, 54, 96], params: 34944 });
    expect(summary.rows[2]).toMatchObject({ name: "dense", outputShape: [null, 8], params: 32776 });
  });
});

describe("torchinfo summaries", () => {
  const text = [
    "==========================================================================================",
    "Layer (type:depth-idx)                   Output Shape              Param #",
    "==========================================================================================",
    "Net                                      [1, 10]                   --",
    "├─Conv2d: 1-1                            [1, 6, 28, 28]            156",
    "├─Sequential: 1-2                        [1, 16, 5, 5]             --",
    "│    └─Conv2d: 2-1                       [1, 16, 10, 10]           2,416",
    "│    └─ReLU: 2-2                         [1, 16, 10, 10]           --",
    "├─Linear: 1-3                            [1, 10]                   850",
    "==========================================================================================",
    "Total params: 3,422",
    "Trainable params: 3,422",
    "Non-trainable params: 0",
    "==========================================================================================",
  ].join("\n");

  it("parses the tree with depths", () => {
    const summaries = parseSummaries(text);
    expect(summaries).toHaveLength(1);
    const summary = summaries[0];
    expect(summary).toMatchObject({ format: "torchinfo", modelName: "Net" });
    expect(summary.totals.total).toBe(3422);
    expect(summary.rows.map((row) => `${row.type}@${row.depth}`)).toEqual([
      "Conv2d@1",
      "Sequential@1",
      "Conv2d@2",
      "ReLU@2",
      "Linear@1",
    ]);
    expect(summary.rows[2]).toMatchObject({ outputShape: [1, 16, 10, 10], params: 2416 });
    expect(summary.rows[1].params).toBeNull();
  });
});

describe("shared helpers", () => {
  it("parses shapes", () => {
    expect(parseShape("(None, 224, 224, 3)")).toEqual([null, 224, 224, 3]);
    expect(parseShape("[(None, 224, 224, 3)]")).toEqual([null, 224, 224, 3]);
    expect(parseShape("(None,)")).toEqual([null]);
    expect(parseShape("--")).toBeNull();
  });
  it("splits names and types", () => {
    expect(splitNameAndType("conv2d_1 (Conv2D)")).toEqual({
      name: "conv2d_1",
      type: "Conv2D",
      truncated: false,
    });
    expect(splitNameAndType("multi_head_attenti… (MultiHeadAttentio…")).toEqual({
      name: "multi_head_attenti",
      type: "MultiHeadAttentio",
      truncated: true,
    });
  });
});

describe("extractOutputFacts", () => {
  it("finds class counts in the reference notebooks' outputs", () => {
    const facts = extractOutputFacts(
      "Found 633 images belonging to 8 classes.\nFound 156 images belonging to 8 classes.",
    );
    expect(facts.classCounts).toEqual([8]);
    expect(
      extractOutputFacts("Number of micro-organism types (classes) found: 8\n - Amoeba")
        .classCounts,
    ).toEqual([8]);
  });
  it("finds printed parameter counts", () => {
    const facts = extractOutputFacts(
      "FireNet: 32.18 M parameters | anchors per image at 224px: 9,261",
    );
    expect(facts.paramMentions).toEqual([{ label: "FireNet", value: 32180000, approximate: true }]);
    expect(extractOutputFacts("Trainable parameters: 1,234,567").paramMentions).toEqual([
      { label: "Trainable parameters", value: 1234567, approximate: false },
    ]);
  });
});
