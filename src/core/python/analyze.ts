import type { Node, Tree } from "web-tree-sitter";
import { evaluateStatic, type Scope, type StaticValue } from "./evaluate";

export type SymbolKind = "assignment" | "function" | "class" | "import";

export type PySymbol = {
  name: string;
  kind: SymbolKind;
  cell: number;
  /** 1-based, inclusive, within the cell. */
  startLine: number;
  endLine: number;
  /** assignment: the right-hand side as written (shortened). */
  valueText: string | null;
  /** assignment: the value when it could be worked out statically. */
  value: StaticValue | undefined;
  /** class: base classes as written, e.g. ["nn.Module"]. */
  bases: string[];
  /** class: method names. function: empty. */
  methods: string[];
  /** import: the module the name comes from, e.g. "tensorflow.keras.layers". */
  module: string | null;
};

export type CellAnalysis = {
  cell: number;
  /** Module-level definitions, in source order. */
  defines: PySymbol[];
  /** Every callee in the cell as written: "Conv2D", "nn.Conv2d", "model.fit". */
  calls: string[];
  /** Attributes assigned anywhere in the cell, e.g. "base_model.trainable". */
  attributeWrites: string[];
  /** Every identifier referenced (for the dependency graph between cells). */
  names: string[];
  /** Modules imported, e.g. ["torch.nn", "tensorflow.keras.layers"]. */
  importedModules: string[];
  statementCount: number;
  hasSyntaxError: boolean;
};

const MAX_VALUE_TEXT = 160;
/** Module-level blocks we look inside for definitions (`if SMOKE: cfg = ...`). */
const TRANSPARENT_BLOCKS = new Set([
  "if_statement",
  "elif_clause",
  "else_clause",
  "try_statement",
  "except_clause",
  "finally_clause",
  "with_statement",
  "block",
]);

/**
 * Extracts definitions, calls and names from one parsed cell.
 * `scope` holds constants from earlier cells and is extended with this cell's
 * constants, so values resolve across cells in notebook order.
 */
export function analyzeCell(
  tree: Tree,
  cell: number,
  scope: Map<string, StaticValue>,
): CellAnalysis {
  const root = tree.rootNode;
  const defines: PySymbol[] = [];
  const importedModules: string[] = [];

  // Inside an `if` / `try` branch we cannot know which assignment runs, so those
  // names are recorded but their values are treated as unknown.
  const visitStatements = (parent: Node, conditional: boolean) => {
    for (const statement of parent.namedChildren) {
      collectStatement(statement, cell, scope, defines, importedModules, conditional);
      if (TRANSPARENT_BLOCKS.has(statement.type)) visitStatements(statement, true);
    }
  };
  visitStatements(root, false);

  const calls: string[] = [];
  for (const call of root.descendantsOfType("call")) {
    const callee = call.childForFieldName("function");
    if (callee && (callee.type === "identifier" || callee.type === "attribute")) {
      const text = callee.text.replace(/\s+/g, "");
      if (text.length <= 120) calls.push(text);
    }
  }

  const attributeWrites: string[] = [];
  for (const assignment of root.descendantsOfType("assignment")) {
    const left = assignment.childForFieldName("left");
    if (left?.type === "attribute") attributeWrites.push(left.text.replace(/\s+/g, ""));
  }

  const names = new Set<string>();
  for (const identifier of root.descendantsOfType("identifier")) names.add(identifier.text);

  return {
    cell,
    defines,
    calls,
    attributeWrites,
    names: [...names],
    importedModules: [...new Set(importedModules)],
    statementCount: root.namedChildren.filter((child) => child.type !== "comment").length,
    hasSyntaxError: root.hasError,
  };
}

function lines(node: Node): { startLine: number; endLine: number } {
  return { startLine: node.startPosition.row + 1, endLine: node.endPosition.row + 1 };
}

function blank(name: string, kind: SymbolKind, cell: number, node: Node): PySymbol {
  return {
    name,
    kind,
    cell,
    ...lines(node),
    valueText: null,
    value: undefined,
    bases: [],
    methods: [],
    module: null,
  };
}

function collectStatement(
  statement: Node,
  cell: number,
  scope: Map<string, StaticValue>,
  defines: PySymbol[],
  importedModules: string[],
  conditional: boolean,
): void {
  let node = statement;
  if (node.type === "decorated_definition") {
    node = node.childForFieldName("definition") ?? node;
  }

  switch (node.type) {
    case "function_definition": {
      const name = node.childForFieldName("name")?.text;
      if (name) defines.push(blank(name, "function", cell, statement));
      return;
    }
    case "class_definition": {
      const name = node.childForFieldName("name")?.text;
      if (!name) return;
      const symbol = blank(name, "class", cell, statement);
      symbol.bases = (node.childForFieldName("superclasses")?.namedChildren ?? [])
        .filter((base) => base.type !== "keyword_argument")
        .map((base) => base.text.replace(/\s+/g, ""));
      symbol.methods = (node.childForFieldName("body")?.namedChildren ?? [])
        .map((member) =>
          member.type === "decorated_definition" ? member.childForFieldName("definition") : member,
        )
        .filter((member): member is Node => member?.type === "function_definition")
        .map((member) => member.childForFieldName("name")?.text ?? "")
        .filter(Boolean);
      defines.push(symbol);
      return;
    }
    case "import_statement":
    case "import_from_statement":
      collectImport(node, cell, defines, importedModules);
      return;
    case "expression_statement": {
      const expression = node.namedChildren[0];
      if (expression?.type === "assignment") {
        collectAssignment(expression, statement, cell, scope, defines, conditional);
      }
      return;
    }
    default:
      return;
  }
}

function collectImport(
  node: Node,
  cell: number,
  defines: PySymbol[],
  importedModules: string[],
): void {
  const fromModule =
    node.type === "import_from_statement"
      ? (node.childForFieldName("module_name")?.text ?? null)
      : null;
  if (fromModule) importedModules.push(fromModule);

  for (const child of node.namedChildren) {
    if (node.type === "import_from_statement" && child.text === fromModule) continue;
    let imported: string | undefined;
    let alias: string | undefined;
    if (child.type === "dotted_name") {
      imported = child.text;
    } else if (child.type === "aliased_import") {
      imported = child.childForFieldName("name")?.text;
      alias = child.childForFieldName("alias")?.text;
    }
    if (!imported) continue;

    if (!fromModule) importedModules.push(imported);
    // `import torch.nn as nn` binds "nn"; `import torch.nn` binds "torch".
    const bound = alias ?? (fromModule ? imported : imported.split(".")[0]);
    const symbol = blank(bound, "import", cell, node);
    symbol.module = fromModule ? `${fromModule}.${imported}` : imported;
    defines.push(symbol);
  }
}

function collectAssignment(
  assignment: Node,
  statement: Node,
  cell: number,
  scope: Map<string, StaticValue>,
  defines: PySymbol[],
  conditional: boolean,
): void {
  const left = assignment.childForFieldName("left");
  const right = assignment.childForFieldName("right");
  if (!left || !right) return;

  // Chained assignment (`a = b = 3`): the right side is another assignment.
  const valueNode = right.type === "assignment" ? right.childForFieldName("right") : right;
  const value = conditional ? undefined : evaluateStatic(valueNode, scope as Scope);
  const valueText = shorten(right.text);

  const targets: Node[] =
    left.type === "pattern_list" || left.type === "tuple_pattern" ? left.namedChildren : [left];

  targets.forEach((target, position) => {
    if (target.type !== "identifier") return;
    const symbol = blank(target.text, "assignment", cell, statement);
    symbol.valueText = valueText;
    // Tuple unpacking: `H, W = 224, 224` gives each name its own element.
    symbol.value =
      targets.length === 1 ? value : Array.isArray(value) ? value[position] : undefined;
    defines.push(symbol);
    if (symbol.value !== undefined) scope.set(symbol.name, symbol.value);
    else scope.delete(symbol.name); // reassigned to something we cannot know
  });
}

function shorten(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > MAX_VALUE_TEXT ? `${flat.slice(0, MAX_VALUE_TEXT - 1)}…` : flat;
}
