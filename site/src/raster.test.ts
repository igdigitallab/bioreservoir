import { describe, expect, it } from "vitest";
import { buildDecisionMath, buildRasterLayout, decodeRasterSpikes, describeRaster } from "./raster";
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

describe("buildRasterLayout", () => {
  function layoutOf(pairs: Array<[number, number]>, totalRows: number, maxPlotHeight = 240) {
    const raster = makeRaster({
      cell_types: Array.from({ length: totalRows }, (_, i) => `n${i}`),
      atlas_indices: Array.from({ length: totalRows }, (_, i) => i),
      spikes_b64: base64OfUint16Pairs(pairs),
    });
    return buildRasterLayout(raster, decodeRasterSpikes(raster), { maxPlotHeight });
  }

  it("drops neurons that never fired and counts them", () => {
    // 10 recorded neurons, only rows 3 and 7 fire.
    const layout = layoutOf(
      [
        [3, 100],
        [7, 200],
        [3, 300],
      ],
      10,
    );
    expect(layout.rows.map((r) => r.row)).toEqual([3, 7]);
    expect(layout.silentRows).toBe(8);
    expect(layout.totalRows).toBe(10);
    expect(layout.spikeCount).toBe(3);
    expect(layout.droppedRows).toBe(0);
  });

  it("sorts busiest first, and breaks ties by neuron index rather than by spike time", () => {
    // Tie-breaking the single-spike rows by time would draw a diagonal across the bottom of the
    // plot that looks like a travelling wave and is only the sort showing through.
    const layout = layoutOf(
      [
        [0, 500], // 1 spike, late
        [1, 100], // 2 spikes
        [1, 400],
        [2, 50], // 1 spike, earliest
      ],
      3,
    );
    expect(layout.rows.map((r) => r.row)).toEqual([1, 0, 2]);
    expect(layout.rows[0]).toMatchObject({ count: 2, firstMs: 10 });
  });

  it("shrinks the row height so the plot fits the height budget", () => {
    const many: Array<[number, number]> = Array.from({ length: 60 }, (_, i) => [i, i * 10]);
    const layout = layoutOf(many, 300, 240);
    expect(layout.rows.length).toBe(60);
    expect(layout.rows.length * layout.rowHeight).toBeLessThanOrEqual(240);
    expect(layout.rowHeight).toBeGreaterThanOrEqual(2);
  });

  it("caps the row height when only a few neurons fired, instead of drawing fat bars", () => {
    const layout = layoutOf(
      [
        [0, 10],
        [1, 20],
      ],
      300,
      240,
    );
    expect(layout.rowHeight).toBe(7);
  });

  it("drops the quietest rows that do not fit even at the minimum row height, and says how many", () => {
    // 200 active rows, budget 100px, minimum row height 2px -> only 50 rows fit.
    const many: Array<[number, number]> = Array.from({ length: 200 }, (_, i) => [i, 10]);
    const layout = layoutOf(many, 300, 100);
    expect(layout.rows.length).toBe(50);
    expect(layout.droppedRows).toBe(150);
    expect(layout.rowHeight).toBe(2);
    expect(describeRaster(layout)).toContain("50 busiest drawn");
  });

  it("says so plainly when nothing fired", () => {
    const layout = layoutOf([], 300);
    expect(layout.rows).toEqual([]);
    expect(describeRaster(layout)).toBe("no spikes from 300 recorded neurons in this trial");
  });

  it("describes what was drawn and what was left out", () => {
    const layout = layoutOf(
      [
        [3, 100],
        [7, 200],
        [3, 300],
      ],
      10,
    );
    expect(describeRaster(layout)).toBe("3 spikes · 2 of 10 recorded neurons fired · silent rows not drawn");
  });
});
