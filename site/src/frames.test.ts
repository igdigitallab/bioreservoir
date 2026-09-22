import { describe, expect, it } from "vitest";
import { buildReplaySchedule, decodeActiveBin, decodeFrames, BIN_DISPLAY_MS } from "./frames";
import type { AnswerFrames } from "./api/types";

function base64OfUint32(values: number[]): string {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((v, i) => view.setUint32(i * 4, v, true));
  let binary = "";
  bytes.forEach((b) => (binary += String.fromCharCode(b)));
  return btoa(binary);
}

describe("decodeActiveBin", () => {
  it("decodes an empty bin to an empty array", () => {
    expect(decodeActiveBin("")).toEqual(new Uint32Array(0));
  });

  it("decodes little-endian uint32 indices regardless of host endianness", () => {
    const b64 = base64OfUint32([0, 1, 65535, 140023, 4294967295]);
    expect(Array.from(decodeActiveBin(b64))).toEqual([0, 1, 65535, 140023, 4294967295]);
  });

  it("rejects a payload that is not a whole number of uint32s", () => {
    // "AAA" decodes to 2 raw bytes, not a multiple of 4.
    expect(() => decodeActiveBin("AAA=")).toThrow();
  });
});

describe("decodeFrames", () => {
  it("decodes all bins in order and matches n_bins", () => {
    const frames: AnswerFrames = {
      bin_ms: 25,
      n_bins: 3,
      active_b64: [base64OfUint32([1, 2]), base64OfUint32([]), base64OfUint32([9])],
    };
    const bins = decodeFrames(frames);
    expect(bins).toHaveLength(3);
    expect(Array.from(bins[0]!)).toEqual([1, 2]);
    expect(Array.from(bins[1]!)).toEqual([]);
    expect(Array.from(bins[2]!)).toEqual([9]);
  });

  it("throws if active_b64 length does not match n_bins", () => {
    const frames: AnswerFrames = { bin_ms: 25, n_bins: 10, active_b64: [base64OfUint32([1])] };
    expect(() => decodeFrames(frames)).toThrow();
  });
});

describe("buildReplaySchedule", () => {
  it("stretches 10 bins of 25ms into a ~2.5s schedule (250ms per bin)", () => {
    const bins = Array.from({ length: 10 }, () => new Uint32Array(0));
    const schedule = buildReplaySchedule(bins);
    expect(BIN_DISPLAY_MS).toBe(250);
    expect(schedule[0]!.atMs).toBe(0);
    expect(schedule[9]!.atMs).toBe(2250);
    // last bin starts at 2250ms and shows for 250ms -> total replay 2.5s.
    expect(schedule[9]!.atMs + BIN_DISPLAY_MS).toBe(2500);
  });
});
