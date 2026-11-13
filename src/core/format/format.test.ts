import { describe, expect, it } from "vitest";
import { formatCompact, formatCount, formatShape } from "./format";

describe("formatCount", () => {
  it("groups thousands (AlexNet total params)", () => {
    expect(formatCount(46781192)).toBe("46,781,192");
  });
  it("handles non-finite values", () => {
    expect(formatCount(Number.NaN)).toBe("–");
  });
});

describe("formatCompact", () => {
  it("uses M for millions", () => {
    expect(formatCompact(46781192)).toBe("46.78 M");
  });
  it("trims trailing zeros", () => {
    expect(formatCompact(2_000_000_000)).toBe("2 G");
  });
  it("leaves small numbers alone", () => {
    expect(formatCompact(704)).toBe("704");
  });
});

describe("formatShape", () => {
  it("joins dims with ×", () => {
    expect(formatShape([54, 54, 96])).toBe("54×54×96");
  });
  it("keeps symbolic dims", () => {
    expect(formatShape(["B", 196, 64])).toBe("B×196×64");
  });
  it("names the empty shape", () => {
    expect(formatShape([])).toBe("scalar");
  });
});
