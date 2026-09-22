import { describe, expect, it } from "vitest";
import { buildDecisionMath, decodeRasterSpikes } from "./raster";
import type { AnswerLab, LabRaster } from "./api/types";

function base64OfUint16Pairs(pairs: Array<[number, number]>): string {
  const bytes = new Uint8Array(pairs.length * 4);
  const view = new DataView(bytes.buffer);
  pairs.forEach(([row, tenthMs], i) => {
    view.setUint16(i * 4, row, true);
    view.setUint16(i * 4 + 2, tenthMs, true);
  });
  let binary = "";
  bytes.forEach((b) => (binary += String.fromCharCode(b)));
  return btoa(binary);
}

function makeRaster(overrides: Partial<LabRaster> = {}): LabRaster {
  return {
    atlas_indices: [10, 20, 30],
    cell_types: ["DNp01", "MN9", "MDN"],
    spikes_b64: "",
    ...overrides,
  };
}

describe("decodeRasterSpikes", () => {
  it("decodes an empty raster to no spikes", () => {
    expect(decodeRasterSpikes(makeRaster())).toEqual([]);
  });

  it("decodes little-endian (row, time in 0.1ms) pairs into ms", () => {
    const b64 = base64OfUint16Pairs([
      [0, 0],
      [1, 15],
      [2, 2500],
    ]);
    const spikes = decodeRasterSpikes(makeRaster({ spikes_b64: b64 }));
    expect(spikes).toEqual([
      { row: 0, timeMs: 0 },
      { row: 1, timeMs: 1.5 },
      { row: 2, timeMs: 250 },
    ]);
  });

  it("rejects a payload that is not a whole number of (row, time) pairs", () => {
    expect(() => decodeRasterSpikes(makeRaster({ spikes_b64: "AAA=" }))).toThrow();
  });
});

function makeLab(overrides: Partial<AnswerLab> = {}): AnswerLab {
  return {
    trials: [
      { seed: 1, spikes_left: 100, spikes_right: 80, bias: 0.111 },
      { seed: 2, spikes_left: 95, spikes_right: 85, bias: 0.056 },
    ],
    b0: 0.02,
    corrected_bias: 0.063,
    yes_side: "left",
    stimulated: { total: 300, left: 150, right: 150, by_modality: {} },
    active_neurons: 12000,
    active_fraction: 0.073,
    total_spikes: 45000,
    readout_latency_ms: null,
    top_cell_types: [],
    raster: makeRaster(),
    provenance: {
      brain: "malecns",
      n_neurons: 165122,
      n_connections: 6235682,
      n_synapses: 89731551,
      min_syn: 5,
      model: "shiu-lif",
      dt_ms: 0.1,
      sim_ms: 250,
      code_sha: "abc123",
      config_hash: "def456",
      reproduce: "uv run python -m bioreservoir.oracle answer --id abc",
    },
    ...overrides,
  };
}

describe("buildDecisionMath", () => {
  it("averages the real per-trial bias values and carries b0/corrected/side through unchanged", () => {
    const math = buildDecisionMath(makeLab());
    expect(math.trialCount).toBe(2);
    expect(math.meanRawBias).toBeCloseTo((0.111 + 0.056) / 2);
    expect(math.b0).toBe(0.02);
    expect(math.correctedBias).toBe(0.063);
    expect(math.yesSide).toBe("left");
  });

  it("does not throw on zero trials (NaN mean, not a fabricated number)", () => {
    const math = buildDecisionMath(makeLab({ trials: [] }));
    expect(math.trialCount).toBe(0);
    expect(Number.isNaN(math.meanRawBias)).toBe(true);
  });
});
