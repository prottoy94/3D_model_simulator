/**
 * The fast path: reads simple layer stacks straight from the code, with no AI.
 * It handles Keras `Sequential([...])` and PyTorch `nn.Sequential(...)` whose
 * layers and arguments can all be resolved statically. Anything it is not sure
 * about is declined with a reason, so a wrong model is never shown as certain.
 */
import type { Node } from "web-tree-sitter";
import { ref, WireBuilder } from "../ir/builder";
import type { CodeRef, ParamValue, WireIR } from "../ir/wire";
import type { NotebookAnalysis, PythonParser } from "../notebook/analyze-notebook";
import type { Notebook } from "../notebook/types";
import { getOp, opForActivation, opForLayerName } from "../ops";
import type { OpKind } from "../ops/kinds";
import { evaluateStatic, type StaticValue } from "../python/evaluate";
import { sanitizePython } from "../python/sanitize";

export type Declined = { cell: number; line: number; reason: string };

export type FastPathResult = {
  /** null when no model could be read with certainty. */
  wire: WireIR | null;
  /** Layer stacks that were found but not understood, and why. */
  declined: Declined[];
};

type Framework = "keras" | "pytorch";

type Layer = {
  op: OpKind;
  layerName: string;
  params: Record<string, ParamValue>;
  raw: Map<string, StaticValue>;
  ref: CodeRef;
};

type Found = {
  name: string;
  framework: Framework;
  inputShape: number[];
  inputAssumed: boolean;
  inputRef: CodeRef;
  layers: Layer[];
  refs: CodeRef[];
};

class Decline extends Error {}

const CLASS_COUNT_NAMES = ["num_classes", "NUM_CLASSES", "n_classes", "N_CLASSES", "num_labels"];
const ASSUMED_IMAGE_SIZE = 224;

const lastSegment = (name: string) => name.slice(name.lastIndexOf(".") + 1);
const lineOf = (node: Node) => node.startPosition.row + 1;
const refOf = (cell: number, node: Node, role: CodeRef["role"] = "declare") =>
  ref(cell, node.startPosition.row + 1, node.endPosition.row + 1, role);
const elements = (node: Node) =>
  node.namedChildren.filter((child): child is Node => child !== null && child.type !== "comment");

function enclosing(node: Node, type: string): Node | null {
  for (let current = node.parent; current; current = current.parent) {
    if (current.type === type) return current;
  }
  return null;
}

function toParamValue(value: StaticValue): ParamValue | undefined {
  if (typeof value === "number" || typeof value === "string" || typeof value === "boolean")
    return value;
  if (Array.isArray(value) && value.every((item): item is number => typeof item === "number"))
    return value;
  return undefined;
}

/** Reads a call's arguments as name → value, using the constructor's positional order. */
function readArgs(
  call: Node,
  positional: string[],
  scope: Map<string, StaticValue>,
): Map<string, StaticValue | undefined> {
  const args = new Map<string, StaticValue | undefined>();
  let index = 0;
  for (const argument of elements(call.childForFieldName("arguments")!)) {
    if (argument.type === "keyword_argument") {
      const name = argument.childForFieldName("name")?.text;
      if (name) args.set(name, evaluateStatic(argument.childForFieldName("value"), scope));
    } else if (argument.type === "list_splat" || argument.type === "dictionary_splat") {
      throw new Decline("uses *args or **kwargs");
    } else {
      args.set(positional[index] ?? `arg${index}`, evaluateStatic(argument, scope));
      index++;
    }
  }
  return args;
}

function layerLabel(layer: Layer): string {
  const { params } = layer;
  const pairText = (value: ParamValue | undefined) =>
    Array.isArray(value) ? value.join("×") : String(value);
  if (layer.op === "conv2d") return `Conv ${params.filters}@${pairText(params.kernel)}`;
  if (layer.op === "dense") {
    const activation =
      typeof params.activation === "string" && params.activation !== "linear"
        ? ` · ${params.activation}`
        : "";
    return `Dense ${params.units}${activation}`;
  }
  if (layer.op === "maxpool2d" || layer.op === "avgpool2d") {
    return `${layer.op === "maxpool2d" ? "MaxPool" : "AvgPool"} ${pairText(params.pool ?? [2, 2])}`;
  }
  if (layer.op === "dropout") return `Dropout ${params.rate ?? 0.5}`;
  return getOp(layer.op)?.title ?? layer.layerName;
}

function readLayer(
  call: Node,
  cell: number,
  framework: Framework,
  scope: Map<string, StaticValue>,
): Layer | { inputShape: number[] } {
  const callee = call.childForFieldName("function")?.text ?? "";
  const layerName = lastSegment(callee);

  if (layerName === "Input" || layerName === "InputLayer") {
    const args = readArgs(call, ["shape"], scope);
    const shape = toParamValue(args.get("shape") ?? args.get("input_shape") ?? null);
    if (!Array.isArray(shape)) throw new Decline("the input shape could not be resolved");
    return { inputShape: shape };
  }

  let op = opForLayerName(layerName, framework);
  if (layerName === "Activation") {
    const name = readArgs(call, ["activation"], scope).get("activation");
    op = typeof name === "string" ? opForActivation(name) : undefined;
    if (!op) throw new Decline(`activation "${String(name)}" is not supported yet`);
    return { op, layerName, params: {}, raw: new Map(), ref: refOf(cell, call) };
  }
  const spec = op ? getOp(op) : undefined;
  const signature = spec?.signatures[framework];
  if (!op || !signature) throw new Decline(`layer "${callee}" is not supported yet`);

  const args = readArgs(call, signature.positional, scope);
  const params: Record<string, ParamValue> = {};
  const raw = new Map<string, StaticValue>();
  for (const [name, value] of args) {
    const key = signature.keys[name];
    const needed =
      key !== undefined ||
      name === "input_shape" ||
      name === "in_channels" ||
      name === "in_features";
    if (value === undefined) {
      if (needed)
        throw new Decline(`"${name}" of ${layerName} could not be resolved from the code`);
      continue;
    }
    raw.set(name, value);
    if (key === undefined || value === null) continue;
    const param = toParamValue(value);
    if (param === undefined)
      throw new Decline(`"${name}" of ${layerName} has an unsupported value`);
    params[key] = param;
  }
  if (op === "global_avg_pool" && framework === "pytorch") {
    const size = raw.get("output_size");
    const isOne = size === 1 || (Array.isArray(size) && size.every((item) => item === 1));
    if (!isOne)
      throw new Decline("AdaptiveAvgPool2d with an output size other than 1 is not supported yet");
  }
  return { op, layerName, params, raw, ref: refOf(cell, call) };
}

/** Binds a function's parameters from its defaults and from one call site. */
function bindFunction(
  fn: Node,
  callSite: Node | null,
  globals: Map<string, StaticValue>,
): Map<string, StaticValue> {
  const scope = new Map(globals);
  const names: string[] = [];
  for (const parameter of elements(fn.childForFieldName("parameters")!)) {
    const nameNode =
      parameter.type === "identifier"
        ? parameter
        : (parameter.childForFieldName("name") ?? parameter.namedChildren[0]);
    if (!nameNode || nameNode.type !== "identifier") continue;
    names.push(nameNode.text);
    scope.delete(nameNode.text); // a parameter hides a global of the same name
    const fallback = evaluateStatic(parameter.childForFieldName("value"), globals);
    if (fallback !== undefined) scope.set(nameNode.text, fallback);
  }
  if (callSite) {
    for (const [name, value] of readArgs(callSite, names, globals)) {
      if (value !== undefined) scope.set(name, value);
    }
  }
  return scope;
}

function readStack(
  call: Node,
  cell: number,
  framework: Framework,
  globals: Map<string, StaticValue>,
  callSites: Map<string, { node: Node; cell: number }>,
  index: number,
): Found {
  const fn = enclosing(call, "function_definition");
  const fnName = fn?.childForFieldName("name")?.text ?? null;
  const site = fnName ? (callSites.get(fnName) ?? null) : null;
  if (fn && !site)
    throw new Decline(`${fnName}() is never called with values that can be resolved`);
  const scope = fn ? bindFunction(fn, site!.node, globals) : globals;

  const args = elements(call.childForFieldName("arguments")!);
  const positional = args.filter((argument) => argument.type !== "keyword_argument");
  const layerNodes = framework === "keras" ? elements(positional[0]!) : positional;
  if (layerNodes.length === 0) throw new Decline("the layer list is empty");

  let inputShape: number[] | null = null;
  let inputRef: CodeRef | null = null;
  const layers: Layer[] = [];
  for (const node of layerNodes) {
    if (node.type !== "call")
      throw new Decline("the layer list contains something that is not a layer call");
    const layer = readLayer(node, cell, framework, scope);
    if ("inputShape" in layer) {
      inputShape = layer.inputShape;
      inputRef = refOf(cell, node);
      continue;
    }
    if (layers.length === 0 && !inputShape) {
      const declared = layer.raw.get("input_shape");
      if (Array.isArray(declared)) {
        inputShape = toParamValue(declared) as number[];
        inputRef = layer.ref;
      }
    }
    layers.push(layer);
  }

  // PyTorch layer stacks do not state their input size, so the first layer has to tell us.
  let inputAssumed = false;
  if (!inputShape && framework === "pytorch") {
    const first = layers[0]!;
    const channels = first.raw.get("in_channels");
    const features = first.raw.get("in_features");
    if (typeof channels === "number")
      inputShape = [channels, ASSUMED_IMAGE_SIZE, ASSUMED_IMAGE_SIZE];
    else if (typeof features === "number") inputShape = [features];
    inputAssumed = inputShape !== null && typeof channels === "number";
    inputRef = first.ref;
  }
  if (!inputShape || !inputRef) throw new Decline("the input shape is not stated in the code");

  const nameArgument = args.find(
    (argument) =>
      argument.type === "keyword_argument" && argument.childForFieldName("name")?.text === "name",
  );
  const named = evaluateStatic(nameArgument?.childForFieldName("value") ?? null, scope);
  const assignment =
    call.parent?.type === "assignment" ? call.parent.childForFieldName("left")?.text : undefined;
  const name =
    (typeof named === "string" && named) ||
    fnName?.replace(/^(build|create|make|get)_/, "") ||
    (assignment && /^[A-Za-z_]\w*$/.test(assignment) ? assignment : `Sequential ${index + 1}`);

  const refs: CodeRef[] = fn ? [refOf(cell, fn, "definition")] : [refOf(cell, call, "definition")];
  if (site) refs.push(refOf(site.cell, site.node, "call"));
  return { name, framework, inputShape, inputAssumed, inputRef, layers, refs };
}

export function extractSequentialModels(
  notebook: Notebook,
  analysis: NotebookAnalysis,
  parser: PythonParser,
): FastPathResult {
  const globals = new Map<string, StaticValue>(Object.entries(analysis.constants));
  // `num_classes = len(class_names)` is only known at run time; the notebook's outputs tell us.
  const classCount = analysis.classCounts[0];
  const seeded: string[] = [];
  if (classCount !== undefined) {
    for (const name of CLASS_COUNT_NAMES) {
      if (!globals.has(name)) {
        globals.set(name, classCount);
        seeded.push(name);
      }
    }
  }

  const trees = notebook.cells
    .filter((cell) => cell.kind === "code")
    .flatMap((cell) => {
      const tree = parser.parse(sanitizePython(cell.source));
      return tree ? [{ cell: cell.index, tree }] : [];
    });

  try {
    // The first call of each function, outside any function body, supplies its argument values.
    const callSites = new Map<string, { node: Node; cell: number }>();
    const stacks: Array<{ node: Node; cell: number; framework: Framework }> = [];
    for (const { cell, tree } of trees) {
      for (const call of tree.rootNode.descendantsOfType("call")) {
        if (!call) continue;
        const callee = call.childForFieldName("function")?.text ?? "";
        if (
          /^[A-Za-z_]\w*$/.test(callee) &&
          !callSites.has(callee) &&
          !enclosing(call, "function_definition")
        ) {
          callSites.set(callee, { node: call, cell });
        }
        if (lastSegment(callee) !== "Sequential" || enclosing(call, "class_definition")) continue;
        const positional = elements(call.childForFieldName("arguments")!).filter(
          (argument) => argument.type !== "keyword_argument",
        );
        if (positional.length === 0) continue; // `Sequential()` followed by .add() calls: not handled yet
        const framework: Framework = positional[0]!.type === "list" ? "keras" : "pytorch";
        stacks.push({ node: call, cell, framework });
      }
    }

    const found: Found[] = [];
    const declined: Declined[] = [];
    for (const stack of stacks) {
      try {
        found.push(
          readStack(stack.node, stack.cell, stack.framework, globals, callSites, found.length),
        );
      } catch (error) {
        if (!(error instanceof Decline)) throw error;
        declined.push({ cell: stack.cell, line: lineOf(stack.node), reason: error.message });
      }
    }
    if (found.length === 0) return { wire: null, declined };

    const frameworks = new Set(found.map((model) => model.framework));
    const b = new WireBuilder({
      title: found.map((model) => model.name).join(" & "),
      summary:
        "Read directly from the code without AI. Every layer and argument was resolved statically.",
      framework: frameworks.size === 1 ? found[0]!.framework : "mixed",
    });
    // Only list the class-count names the notebook's code actually uses.
    const code = notebook.cells
      .filter((cell) => cell.kind === "code")
      .map((cell) => cell.source)
      .join("\n");
    for (const name of seeded.filter((candidate) => new RegExp(`\\b${candidate}\\b`).test(code))) {
      b.symbol(name, classCount, {
        provenance: "from-output",
        note: "number of classes reported in the notebook's outputs",
      });
    }

    const usedIds = new Set<string>();
    for (const model of found) {
      let slug =
        model.name
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "_")
          .replace(/^_|_$/g, "") || "model";
      while (usedIds.has(slug)) slug += "_2";
      usedIds.add(slug);

      b.model(`m_${slug}`, model.name, "", model.refs);
      if (model.inputAssumed) {
        b.symbol(`${slug}_image_size`, ASSUMED_IMAGE_SIZE, {
          provenance: "assumed",
          note: `${model.name} does not state its input size; ${ASSUMED_IMAGE_SIZE}×${ASSUMED_IMAGE_SIZE} is assumed`,
        });
        b.warn(
          `${model.name}: the input image size is not in the code, so ${ASSUMED_IMAGE_SIZE}×${ASSUMED_IMAGE_SIZE} is assumed.`,
        );
      }
      let previous = `n_${slug}_in`;
      b.node(
        previous,
        "input",
        { shape: model.inputShape },
        {
          label: `Input ${model.inputShape.join("×")}`,
          refs: [model.inputRef],
          trainable: "n/a",
          confidence: model.inputAssumed ? "low" : "high",
        },
      );
      model.layers.forEach((layer, position) => {
        const id = `n_${slug}_${position + 1}`;
        const hasWeights = [
          "conv2d",
          "depthwise_conv2d",
          "dense",
          "batchnorm",
          "layernorm",
          "embedding",
          "multi_head_attention",
        ].includes(layer.op);
        b.node(id, layer.op, layer.params, {
          from: [previous],
          label: layerLabel(layer),
          refs: [layer.ref],
          trainable: hasWeights ? "yes" : "n/a",
        });
        previous = id;
      });
      b.outputs(previous);
    }
    return { wire: b.build(), declined };
  } finally {
    for (const { tree } of trees) tree.delete();
  }
}
