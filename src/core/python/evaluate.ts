import type { Node } from "web-tree-sitter";

/** A Python value we can know statically: constants and simple arithmetic on them. */
export type StaticValue = number | string | boolean | null | StaticValue[];

export type Scope = ReadonlyMap<string, StaticValue>;

/**
 * Evaluates a small, safe subset of Python expressions: literals, tuples/lists,
 * arithmetic, names already known, and `len()` / `int()` / `float()` / `max()` /
 * `min()` of those. Returns undefined for anything else. Nothing is executed.
 *
 * This is what turns `NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) ** 2` into 196.
 */
export function evaluateStatic(node: Node | null, scope: Scope): StaticValue | undefined {
  if (!node) return undefined;
  switch (node.type) {
    case "integer": {
      const value = Number(node.text.replace(/_/g, ""));
      return Number.isFinite(value) ? value : undefined;
    }
    case "float": {
      const value = Number(node.text.replace(/_/g, ""));
      return Number.isFinite(value) ? value : undefined;
    }
    case "true":
      return true;
    case "false":
      return false;
    case "none":
      return null;
    case "string":
      return stringValue(node);
    case "identifier":
      return scope.get(node.text);
    case "parenthesized_expression":
      return evaluateStatic(node.namedChildren[0] ?? null, scope);
    case "tuple":
    case "expression_list": // bare `224, 224` on the right of a tuple assignment
    case "list": {
      const items: StaticValue[] = [];
      for (const child of node.namedChildren) {
        const value = evaluateStatic(child, scope);
        if (value === undefined) return undefined;
        items.push(value);
      }
      return items;
    }
    case "unary_operator": {
      const operand = evaluateStatic(node.childForFieldName("argument"), scope);
      if (typeof operand !== "number") return undefined;
      const operator = node.childForFieldName("operator")?.text;
      if (operator === "-") return -operand;
      if (operator === "+") return operand;
      return undefined;
    }
    case "binary_operator":
      return binary(
        node.childForFieldName("operator")?.text,
        evaluateStatic(node.childForFieldName("left"), scope),
        evaluateStatic(node.childForFieldName("right"), scope),
      );
    case "call":
      return call(node, scope);
    default:
      return undefined;
  }
}

function stringValue(node: Node): string | undefined {
  // f-strings and other prefixed strings with interpolation are not constants.
  if (node.descendantsOfType("interpolation").length > 0) return undefined;
  const content = node.namedChildren.filter((child) => child.type === "string_content");
  return content.map((child) => child.text).join("");
}

function binary(
  operator: string | undefined,
  left: StaticValue | undefined,
  right: StaticValue | undefined,
): StaticValue | undefined {
  if (typeof left === "string" && typeof right === "string" && operator === "+") {
    return left + right;
  }
  if (Array.isArray(left) && Array.isArray(right) && operator === "+") return [...left, ...right];
  if (typeof left !== "number" || typeof right !== "number") return undefined;
  let result: number;
  switch (operator) {
    case "+":
      result = left + right;
      break;
    case "-":
      result = left - right;
      break;
    case "*":
      result = left * right;
      break;
    case "/":
      if (right === 0) return undefined;
      result = left / right;
      break;
    case "//":
      if (right === 0) return undefined;
      result = Math.floor(left / right);
      break;
    case "%":
      if (right === 0) return undefined;
      result = left - right * Math.floor(left / right); // Python modulo follows the divisor's sign
      break;
    case "**":
      result = left ** right;
      break;
    default:
      return undefined;
  }
  return Number.isFinite(result) ? result : undefined;
}

function call(node: Node, scope: Scope): StaticValue | undefined {
  const callee = node.childForFieldName("function");
  const argumentList = node.childForFieldName("arguments");
  if (!callee || callee.type !== "identifier" || !argumentList) return undefined;
  const values: StaticValue[] = [];
  for (const argument of argumentList.namedChildren) {
    const value = evaluateStatic(argument, scope);
    if (value === undefined) return undefined; // also rejects keyword arguments
    values.push(value);
  }
  const [first] = values;
  switch (callee.text) {
    case "len":
      return values.length === 1 && (Array.isArray(first) || typeof first === "string")
        ? first.length
        : undefined;
    case "int":
      return values.length === 1 && typeof first === "number" ? Math.trunc(first) : undefined;
    case "float":
      return values.length === 1 && typeof first === "number" ? first : undefined;
    case "max":
    case "min": {
      const numbers = values.length === 1 && Array.isArray(first) ? first : values;
      if (numbers.length === 0 || !numbers.every((n): n is number => typeof n === "number")) {
        return undefined;
      }
      return callee.text === "max" ? Math.max(...numbers) : Math.min(...numbers);
    }
    default:
      return undefined;
  }
}

/** Renders a static value the way Python would print it, e.g. "(224, 224, 3)". */
export function formatStaticValue(value: StaticValue): string {
  if (value === null) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `(${value.map(formatStaticValue).join(", ")})`;
  return String(value);
}
