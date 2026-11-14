import type { CellAnalysis } from "../python/analyze";

/**
 * What a cell is for. Only `model` cells are simulated; the rest give context
 * (see CLAUDE.md, "Scope of the simulation").
 */
export type CellRole =
  | "markdown"
  | "setup"
  | "imports"
  | "config"
  | "data"
  | "model"
  | "compile"
  | "train"
  | "eval"
  | "viz"
  | "other";

export type RoleResult = {
  role: CellRole;
  confidence: "high" | "medium" | "low";
  /** Human-readable reasons, shown in the Notebook Map tooltip. */
  signals: string[];
};

/** Layer constructors: seeing several of these means a cell builds architecture. */
const LAYER_NAMES = new Set([
  // Keras
  "Conv1D",
  "Conv2D",
  "Conv3D",
  "Conv2DTranspose",
  "DepthwiseConv2D",
  "SeparableConv2D",
  "Dense",
  "Flatten",
  "Dropout",
  "BatchNormalization",
  "LayerNormalization",
  "GroupNormalization",
  "MaxPooling1D",
  "MaxPooling2D",
  "AveragePooling2D",
  "GlobalAveragePooling1D",
  "GlobalAveragePooling2D",
  "GlobalMaxPooling2D",
  "Embedding",
  "LSTM",
  "GRU",
  "SimpleRNN",
  "Bidirectional",
  "MultiHeadAttention",
  "Add",
  "Concatenate",
  "Multiply",
  "Reshape",
  "Lambda",
  "Activation",
  "ReLU",
  "Input",
  "InputLayer",
  "TimeDistributed",
  "UpSampling2D",
  "ZeroPadding2D",
  "Sequential",
  "Model",
  // PyTorch
  "Conv1d",
  "Conv2d",
  "Conv3d",
  "ConvTranspose2d",
  "Linear",
  "BatchNorm1d",
  "BatchNorm2d",
  "LayerNorm",
  "GroupNorm",
  "InstanceNorm2d",
  "MaxPool2d",
  "AvgPool2d",
  "AdaptiveAvgPool2d",
  "AdaptiveMaxPool2d",
  "GELU",
  "SiLU",
  "Sigmoid",
  "Tanh",
  "Softmax",
  "LeakyReLU",
  "Identity",
  "ModuleList",
  "ModuleDict",
  "TransformerEncoder",
  "TransformerEncoderLayer",
  "MultiheadAttention",
  "Upsample",
  "PixelShuffle",
  "Parameter",
]);

/** Base classes that make a class definition part of the architecture. */
const MODULE_BASE = /(^|\.)(Module|Model|Layer|LightningModule|PreTrainedModel)$/;

/** Calls that load a ready-made (usually pretrained) network. */
const BACKBONE_CALL =
  /(^|\.)(from_preset|from_pretrained|create_model)$|(^|\.)(applications|torchvision\.models|models)\.[A-Za-z_]\w*$|^(EfficientNet(V2)?[A-Z]\w*|MobileNet(V[23])?\w*|ResNet\d+\w*|VGG1[69]|DenseNet\d+|Inception\w*|Xception|NASNet\w*|ConvNeXt\w*|resnet\d+|vgg1[69](_bn)?|convnext_\w+|efficientnet_\w+|mobilenet_\w+|densenet\d+|vit_[bl]_\d+|swin_\w+)$/;

/** Loaders that appear just as often in inference or benchmark cells, so they count for less. */
const WEAK_BACKBONE_CALL = /(^|\.)(YOLO|load_model|load_state_dict)$/;

/** scikit-learn style estimators (classical ML models). */
const ESTIMATOR_CALL =
  /^(\w+\.)*(RandomForest|GradientBoosting|DecisionTree|ExtraTrees|AdaBoost|HistGradientBoosting|XGB|LGBM|CatBoost)(Classifier|Regressor)$|^(\w+\.)*(LogisticRegression|LinearRegression|Ridge|Lasso|ElasticNet|SVC|SVR|LinearSVC|KNeighbors(Classifier|Regressor)|KMeans|DBSCAN|PCA|GaussianNB|MultinomialNB|MLP(Classifier|Regressor))$/;

const DATA_CALL =
  /(^|\.)(flow_from_directory|flow_from_dataframe|image_dataset_from_directory|ImageDataGenerator|DataLoader|read_csv|read_excel|read_json|load_dataset|train_test_split|mount|load_img|imread|ImageFolder|random_split|fetch_\w+|load_iris|load_digits|listdir|iterdir|glob|rglob|compute_class_weight)$/;

const TRAIN_CALL =
  /(^|\.)(fit|fit_generator|train_on_batch|backward|train_one_epoch)$|^trainer\.train$/;
const OPTIMIZER_STEP = /(^|\.)(step|zero_grad)$/;
const COMPILE_CALL = /(^|\.)(compile|summary)$|^(torchinfo\.)?summary$/;
const TRAINING_SETUP_CALL =
  /(^|\.)(EarlyStopping|ModelCheckpoint|ReduceLROnPlateau|LearningRateScheduler|TensorBoard|CSVLogger|Adam|AdamW|SGD|RMSprop|CosineAnnealingLR|OneCycleLR|StepLR|GradScaler)$/;
const EVAL_CALL =
  /(^|\.)(evaluate|predict|predict_proba|predict_classes|eval|no_grad|inference_mode|FlopCounterMode|perf_counter|classification_report|confusion_matrix|accuracy_score|f1_score|precision_score|recall_score|roc_auc_score|roc_curve|score|COCOeval)$/;
const VIZ_CALL =
  /^(plt|sns|ax|axes|fig|px|go)\b|(^|\.)(imshow|subplot|subplots|plot|bar|hist|scatter|show|savefig|heatmap|plot_model)$/;

const last = (callee: string) => callee.slice(callee.lastIndexOf(".") + 1);

/** Scores every role from the cell's static analysis and picks the strongest. */
export function classifyCodeCell(analysis: CellAnalysis, source: string): RoleResult {
  const { calls, defines, attributeWrites } = analysis;
  const scores: Record<Exclude<CellRole, "markdown">, number> = {
    setup: 0,
    imports: 0,
    config: 0,
    data: 0,
    model: 0,
    compile: 0,
    train: 0,
    eval: 0,
    viz: 0,
    other: 0,
  };
  const signals: string[] = [];
  const note = (role: keyof typeof scores, points: number, reason: string) => {
    scores[role] += points;
    signals.push(reason);
  };

  // ── model ──
  const moduleClasses = defines.filter(
    (symbol) => symbol.kind === "class" && symbol.bases.some((base) => MODULE_BASE.test(base)),
  );
  if (moduleClasses.length > 0) {
    note(
      "model",
      5 * moduleClasses.length,
      `defines ${moduleClasses.map((c) => c.name).join(", ")}`,
    );
  }
  const layerCalls = calls.filter((callee) => LAYER_NAMES.has(last(callee)));
  if (layerCalls.length >= 2) {
    note("model", Math.min(8, layerCalls.length), `${layerCalls.length} layer constructors`);
  }
  const backbones = calls.filter((callee) => BACKBONE_CALL.test(callee));
  if (backbones.length > 0) note("model", 4, `loads ${last(backbones[0])}`);
  const weakBackbones = calls.filter((callee) => WEAK_BACKBONE_CALL.test(callee));
  if (weakBackbones.length > 0) note("model", 2, `loads ${last(weakBackbones[0])}`);
  const estimators = calls.filter((callee) => ESTIMATOR_CALL.test(callee));
  if (estimators.length > 0) note("model", 4, `creates ${last(estimators[0])}`);

  // ── compile / training setup ──
  const compileCalls = calls.filter((callee) => COMPILE_CALL.test(callee));
  if (compileCalls.length > 0) note("compile", 3, `calls ${last(compileCalls[0])}()`);
  if (attributeWrites.some((target) => /\.(trainable|requires_grad)$/.test(target))) {
    note("compile", 2, "freezes or unfreezes layers");
  }

  const setupCalls = calls.filter((callee) => TRAINING_SETUP_CALL.test(callee));
  if (setupCalls.length > 0) note("compile", 2, `creates ${last(setupCalls[0])}`);

  // ── train ──
  const trainCalls = calls.filter((callee) => TRAIN_CALL.test(callee));
  if (trainCalls.length > 0) note("train", 5, `calls ${last(trainCalls[0])}()`);
  if (calls.some((callee) => OPTIMIZER_STEP.test(callee)) && trainCalls.length > 0) {
    note("train", 2, "optimizer loop");
  }
  if (calls.some((callee) => /(^|\.)train$/.test(callee) && !/^model\.train$/.test(callee))) {
    note("train", 2, "calls train()");
  }

  // ── eval ──
  const evalCalls = calls.filter((callee) => EVAL_CALL.test(callee));
  if (evalCalls.length > 0) {
    note("eval", 2 + Math.min(3, evalCalls.length), `calls ${last(evalCalls[0])}()`);
  }

  // ── data ──
  const dataCalls = calls.filter((callee) => DATA_CALL.test(callee));
  if (dataCalls.length > 0) {
    note("data", 2 + Math.min(3, dataCalls.length), `calls ${last(dataCalls[0])}()`);
  }
  const datasetClasses = defines.filter(
    (symbol) => symbol.kind === "class" && symbol.bases.some((base) => /Dataset$/.test(base)),
  );
  if (datasetClasses.length > 0) note("data", 5, `defines dataset ${datasetClasses[0].name}`);

  // ── viz ──
  const vizCalls = calls.filter((callee) => VIZ_CALL.test(callee));
  if (vizCalls.length >= 2) note("viz", Math.min(5, 1 + vizCalls.length / 2), "plots figures");

  // ── imports / config / setup ──
  const statements = Math.max(1, analysis.statementCount);
  const importCount = defines.filter((symbol) => symbol.kind === "import").length;
  const importStatements = (source.match(/^\s*(import|from)\s+\w/gm) ?? []).length;
  if (importStatements > 0 && importStatements / statements >= 0.5) {
    note("imports", 3 + Math.min(2, importCount / 5), "mostly imports");
  }
  const constants = defines.filter(
    (symbol) => symbol.kind === "assignment" && /^[A-Z][A-Z0-9_]*$/.test(symbol.name),
  );
  if (constants.length >= 2 && constants.length / statements >= 0.4) {
    note("config", 3 + Math.min(2, constants.length / 4), `${constants.length} constants`);
  }
  const magicLines = (source.match(/^\s*[!%]/gm) ?? []).length;
  const codeLines = source
    .split("\n")
    .filter((line) => line.trim() && !line.trim().startsWith("#"));
  if (magicLines > 0 && magicLines >= codeLines.length / 2) {
    note("setup", 4, "shell or magic commands");
  }

  // Pick the winner. On a tie the earlier role in this list wins.
  const order: Array<keyof typeof scores> = [
    "model",
    "train",
    "compile",
    "eval",
    "data",
    "viz",
    "config",
    "imports",
    "setup",
  ];
  let best: keyof typeof scores = "other";
  let bestScore = 0;
  let runnerUp = 0;
  for (const role of order) {
    if (scores[role] > bestScore) {
      runnerUp = bestScore;
      best = role;
      bestScore = scores[role];
    } else if (scores[role] > runnerUp) {
      runnerUp = scores[role];
    }
  }
  if (bestScore === 0) return { role: "other", confidence: "low", signals: [] };

  const margin = bestScore - runnerUp;
  return {
    role: best,
    confidence: bestScore >= 5 && margin >= 2 ? "high" : margin >= 1 ? "medium" : "low",
    signals,
  };
}

/** Frameworks a notebook uses, from its imports. */
export function detectFrameworks(importedModules: readonly string[]): string[] {
  const table: Array<[string, RegExp]> = [
    ["pytorch", /^(torch|torchvision|pytorch_lightning|lightning|timm)(\.|$)/],
    ["keras", /^(tensorflow\.keras|keras|keras_hub|keras_cv|keras_nlp)(\.|$)/],
    ["tensorflow", /^tensorflow(\.|$)/],
    ["sklearn", /^(sklearn|xgboost|lightgbm|catboost)(\.|$)/],
    ["huggingface", /^(transformers|diffusers|datasets)(\.|$)/],
    ["ultralytics", /^ultralytics(\.|$)/],
    ["jax", /^(jax|flax|haiku)(\.|$)/],
  ];
  return table
    .filter(([, pattern]) => importedModules.some((module) => pattern.test(module)))
    .map(([name]) => name);
}
