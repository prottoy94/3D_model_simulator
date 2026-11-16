/**
 * Zoo facts for pretrained backbones: output shape and parameter counts of the
 * feature extractor (no classifier head), so a model built on one gets exact
 * totals while the backbone stays one collapsed block.
 *
 * Full layer-by-layer templates (doc 03 §5) come later; these facts are the
 * honest minimum. `verified` says where the numbers come from.
 */

export type ZooEntry = {
  key: string;
  title: string;
  /** Names as written in code, matched case-insensitively on the last segment. */
  aliases: string[];
  framework: "keras" | "pytorch";
  /** "image": (H/stride, W/stride, channels). "tokens": (tokens, width). */
  output:
    | { kind: "image"; stride: number; channels: number }
    | { kind: "tokens"; patch: number; width: number; classToken: boolean };
  /** Intermediate feature maps a model may read (stage 1 first): stride and channels of each. */
  stages?: Array<{ stride: number; channels: number }>;
  /** All weights of the feature extractor, including BatchNorm statistics in Keras. */
  params: number;
  /** Weights that are never trained (Keras BatchNorm moving statistics). */
  nonTrainable: number;
  /** "summary": equals a saved model.summary() in the reference notebooks. "library": from the library's documentation. */
  verified: "summary" | "library";
  description: string;
};

export const ZOO: ZooEntry[] = [
  {
    key: "keras.applications.EfficientNetB0",
    title: "EfficientNetB0",
    aliases: ["EfficientNetB0", "efficientnet_b0", "efficientnetb0"],
    framework: "keras",
    output: { kind: "image", stride: 32, channels: 1280 },
    params: 4_049_571,
    nonTrainable: 42_023,
    verified: "summary",
    description:
      "EfficientNetB0 without its classifier: a stem convolution, seven stages of MBConv blocks (depthwise convolutions with squeeze-and-excite) and a final 1×1 convolution to 1280 channels.",
  },
  {
    key: "keras.applications.MobileNetV2",
    title: "MobileNetV2",
    aliases: ["MobileNetV2", "mobilenet_v2", "mobilenetv2"],
    framework: "keras",
    output: { kind: "image", stride: 32, channels: 1280 },
    params: 2_257_984,
    nonTrainable: 34_112,
    verified: "summary",
    description:
      "MobileNetV2 without its classifier: inverted residual blocks built from cheap depthwise convolutions, ending in a 1×1 convolution to 1280 channels.",
  },
  {
    key: "keras.applications.ResNet50",
    title: "ResNet50",
    aliases: ["ResNet50"],
    framework: "keras",
    output: { kind: "image", stride: 32, channels: 2048 },
    params: 23_587_712,
    nonTrainable: 53_120,
    verified: "library",
    description:
      "ResNet50 without its classifier: four stages of bottleneck blocks with residual connections.",
  },
  {
    key: "keras.applications.VGG16",
    title: "VGG16",
    aliases: ["VGG16"],
    framework: "keras",
    output: { kind: "image", stride: 32, channels: 512 },
    params: 14_714_688,
    nonTrainable: 0,
    verified: "library",
    description: "VGG16 without its dense layers: thirteen 3×3 convolutions in five blocks.",
  },
  {
    key: "keras_hub.ViTBackbone:vit_base_patch16_224_imagenet",
    title: "ViT-B/16",
    aliases: ["ViTBackbone", "vit_base_patch16_224_imagenet", "vit_base_patch16_224", "vit_b_16"],
    framework: "keras",
    output: { kind: "tokens", patch: 16, width: 768, classToken: true },
    params: 85_798_656,
    nonTrainable: 0,
    verified: "summary",
    description:
      "ViT-Base with 16×16 patches: the image becomes 196 patch tokens plus one class token, which pass through 12 Transformer encoder blocks of width 768.",
  },
  {
    key: "torchvision.models.convnext_tiny",
    title: "ConvNeXt-Tiny",
    aliases: ["convnext_tiny", "ConvNeXt_Tiny"],
    framework: "pytorch",
    output: { kind: "image", stride: 32, channels: 768 },
    stages: [
      { stride: 4, channels: 96 },
      { stride: 8, channels: 192 },
      { stride: 16, channels: 384 },
      { stride: 32, channels: 768 },
    ],
    params: 27_818_592,
    nonTrainable: 0,
    verified: "library",
    description:
      "ConvNeXt-Tiny features: a 4×4 patchify stem and four stages (96, 192, 384, 768 channels) of 7×7 depthwise blocks.",
  },
  {
    key: "torchvision.models.resnet50",
    title: "ResNet50",
    aliases: ["resnet50"],
    framework: "pytorch",
    output: { kind: "image", stride: 32, channels: 2048 },
    params: 23_508_032,
    nonTrainable: 0,
    verified: "library",
    description:
      "ResNet50 without its final linear layer: four stages of bottleneck blocks with residual connections.",
  },
];

const plain = (text: string) => text.toLowerCase().replace(/[^a-z0-9]/g, "");

export function findZooEntry(name: string, framework?: "keras" | "pytorch"): ZooEntry | undefined {
  const wanted = plain(name.slice(name.lastIndexOf(".") + 1));
  const full = plain(name);
  const matches = ZOO.filter(
    (entry) =>
      plain(entry.key) === full ||
      full.endsWith(plain(entry.key)) ||
      entry.aliases.some((alias) => plain(alias) === wanted || full.includes(plain(alias))),
  );
  return matches.find((entry) => entry.framework === framework) ?? matches[0];
}

export type ZooFacts = {
  entry: ZooEntry;
  /** Output shape without the batch, in the framework's native layout. */
  outputShape: number[];
  paramCount: number;
  nonTrainableCount: number;
};

/** Facts for a backbone fed with images of the given size (height, width). */
/**
 * The feature map after one stage of a backbone (1 = the first stage). The
 * weights are counted on the backbone node itself, so a tap has none.
 */
export function resolveZooStage(
  name: string,
  stage: number,
  imageSize: [number, number] = [224, 224],
  framework?: "keras" | "pytorch",
): { entry: ZooEntry; outputShape: number[] } | null {
  const entry = findZooEntry(name, framework);
  const info = entry?.stages?.[stage - 1];
  if (!entry || !info) return null;
  const h = Math.ceil(imageSize[0] / info.stride);
  const w = Math.ceil(imageSize[1] / info.stride);
  return {
    entry,
    outputShape: entry.framework === "pytorch" ? [info.channels, h, w] : [h, w, info.channels],
  };
}

export function resolveZoo(
  name: string,
  imageSize: [number, number] = [224, 224],
  framework?: "keras" | "pytorch",
): ZooFacts | null {
  const entry = findZooEntry(name, framework);
  if (!entry) return null;
  const [h, w] = imageSize;
  let outputShape: number[];
  if (entry.output.kind === "tokens") {
    const tokens = Math.floor(h / entry.output.patch) * Math.floor(w / entry.output.patch);
    outputShape = [tokens + (entry.output.classToken ? 1 : 0), entry.output.width];
  } else {
    const oh = Math.ceil(h / entry.output.stride);
    const ow = Math.ceil(w / entry.output.stride);
    outputShape =
      entry.framework === "pytorch"
        ? [entry.output.channels, oh, ow]
        : [oh, ow, entry.output.channels];
  }
  return {
    entry,
    outputShape,
    paramCount: entry.params - entry.nonTrainable,
    nonTrainableCount: entry.nonTrainable,
  };
}

export const ZOO_NAMES: string[] = ZOO.map((entry) => entry.title);
