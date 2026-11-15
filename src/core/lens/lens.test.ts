import { describe, expect, it } from "vitest";
import { buildRuntimeIR } from "../ir/build";
import { alexnetIR, miniVitIR } from "../ir/examples";
import type { RuntimeIR } from "../ir/runtime";
import { describeFocus, resolveSelection, type LensInterval } from "./index";

function build(wire: unknown): RuntimeIR {
  const result = buildRuntimeIR(wire);
  if (!result.ok) throw new Error(result.errors.join("\n"));
  return result.ir;
}

const vit = build(miniVitIR());
const alexnet = build(alexnetIR());
/** Lines of cell 16 of the ViT notebook, which the example IR points at. */
const lines = (startLine: number, endLine = startLine) => ({
  ranges: [{ cell: 16, startLine, endLine }],
});

describe("selecting a block's definition", () => {
  const focus = resolveSelection(vit, "m_vit", lines(39, 58))!;

  it("isolates one copy of the repeated block and remembers the others", () => {
    expect(focus.kind).toBe("block");
    expect(focus.label).toBe("Transformer block");
    expect(focus.copies).toBe(4);
    expect(focus.nodeIds).toHaveLength(9);
    expect(focus.nodeIds.every((id) => id.endsWith("#1"))).toBe(true);
    expect(focus.allNodeIds).toHaveLength(36);
    expect(describeFocus(focus)).toBe("Transformer block · one of 4 copies · 9 layers");
  });

  it("finds what feeds the block and what reads from it", () => {
    expect(focus.inputs).toEqual(["n_pos"]);
    expect(focus.outputs).toEqual(["n_ln1#2", "n_add1#2"]);
  });
});

describe("other selections", () => {
  it("a loop selects every repetition", () => {
    const focus = resolveSelection(vit, "m_vit", lines(68, 69))!;
    expect(focus.kind).toBe("group");
    expect(focus.groupId).toBe("g_blocks");
    expect(focus.nodeIds).toHaveLength(36);
    expect(focus.copies).toBe(4);
    expect(focus.inputs).toEqual(["n_pos"]);
    expect(focus.outputs).toEqual(["n_ln_out"]);
  });

  it("one line inside a block selects that layer in the first copy and highlights all copies", () => {
    const focus = resolveSelection(vit, "m_vit", lines(52))!;
    expect(focus.kind).toBe("layers");
    expect(focus.nodeIds).toEqual(["n_mlp1#1"]);
    expect(focus.allNodeIds).toEqual(["n_mlp1#1", "n_mlp1#2", "n_mlp1#3", "n_mlp1#4"]);
    expect(focus.copies).toBe(4);
    expect(describeFocus(focus)).toBe("MLP Dense 128 · GELU · in each of 4 copies");
  });

  it("several lines select the layers they create", () => {
    const focus = resolveSelection(vit, "m_vit", lines(44, 48))!;
    expect(focus.nodeIds).toEqual(["n_ln1#1", "n_mha#1", "n_add1#1"]);
    expect(focus.inputs).toEqual(["n_pos"]);
  });

  it("a custom layer's class body selects its group", () => {
    const focus = resolveSelection(vit, "m_vit", lines(23, 36))!;
    expect(focus.kind).toBe("group");
    expect(focus.label).toBe("PatchEncoder");
    expect(focus.nodeIds).toEqual(["n_proj", "n_pos"]);
    expect(focus.copies).toBe(1);
  });

  it("two separate lines also take the layers between them", () => {
    const focus = resolveSelection(alexnet, "m_alexnet", {
      ranges: [
        { cell: 14, startLine: 13, endLine: 13 },
        { cell: 14, startLine: 19, endLine: 19 },
      ],
    })!;
    expect(focus.nodeIds).toEqual(["n_bn1", "n_pool1", "n_conv2", "n_bn2", "n_pool2"]);
    expect(focus.inputs).toEqual(["n_conv1"]);
    expect(focus.outputs).toEqual(["n_conv3"]);
  });

  it("returns nothing for code that defines no layer", () => {
    expect(
      resolveSelection(vit, "m_vit", { ranges: [{ cell: 3, startLine: 1, endLine: 4 }] }),
    ).toBeNull();
    expect(resolveSelection(vit, "m_vit", lines(59, 60))).toBeNull();
    expect(resolveSelection(vit, "m_other", lines(39, 58))).toBeNull();
  });
});

describe("selecting a constant", () => {
  const elsewhere = [{ cell: 4, startLine: 3, endLine: 3 }];

  it("PATCH_SIZE lights up the layer that cuts the image into patches", () => {
    const focus = resolveSelection(vit, "m_vit", { ranges: elsewhere, symbols: ["PATCH_SIZE"] })!;
    expect(focus.kind).toBe("symbol");
    expect(focus.nodeIds).toEqual(["n_patches"]);
    expect(describeFocus(focus)).toBe("1 layer depends on PATCH_SIZE");
  });

  it("PROJECTION_DIM reaches the projection, the positions, and every block", () => {
    const focus = resolveSelection(vit, "m_vit", {
      ranges: elsewhere,
      symbols: ["PROJECTION_DIM"],
    })!;
    expect(focus.nodeIds).toEqual(
      expect.arrayContaining(["n_proj", "n_pos", "n_mha#1", "n_mlp2#4"]),
    );
    expect(focus.nodeIds).not.toContain("n_mlp1#1"); // that one uses MLP_DIM
    expect(focus.nodeIds).toHaveLength(2 + 4 * 2);
  });

  it("the loop count selects everything the loop repeats", () => {
    const focus = resolveSelection(vit, "m_vit", {
      ranges: elsewhere,
      symbols: ["TRANSFORMER_LAYERS"],
    })!;
    expect(focus.nodeIds).toHaveLength(36);
  });

  it("ignores names that are not symbols of the model", () => {
    expect(resolveSelection(vit, "m_vit", { ranges: elsewhere, symbols: ["EPOCHS"] })).toBeNull();
  });
});

describe("custom code locations", () => {
  it("works on a generated listing, where every layer has its own line", () => {
    const intervals: LensInterval[] = [
      { cell: -1, startLine: 5, endLine: 5, role: "declare", nodeId: "n_conv1" },
      { cell: -1, startLine: 6, endLine: 6, role: "declare", nodeId: "n_bn1" },
      { cell: -1, startLine: 5, endLine: 7, role: "definition", groupId: "g_block1" },
    ];
    const one = resolveSelection(alexnet, "m_alexnet", {
      ranges: [{ cell: -1, startLine: 6, endLine: 6 }],
      intervals,
    })!;
    expect(one.nodeIds).toEqual(["n_bn1"]);
    const group = resolveSelection(alexnet, "m_alexnet", {
      ranges: [{ cell: -1, startLine: 4, endLine: 8 }],
      intervals,
    })!;
    expect(group.kind).toBe("group");
    expect(group.nodeIds).toEqual(["n_conv1", "n_bn1", "n_pool1"]);
  });
});
