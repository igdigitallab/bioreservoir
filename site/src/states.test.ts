import { describe, expect, it } from "vitest";
import { stateToAnimations } from "./states";
import type { Answer } from "./api/types";

function makeAnswer(overrides: Partial<Answer> = {}): Answer {
  return {
    id: "abc123",
    question: "Will the sun rise tomorrow?",
    answer: "yes",
    confidence: 0.8,
    lateral_bias: 0.12,
    states: {
      appetite: null,
      fear: null,
      backoff: null,
      courtship: null,
      arousal: 0.4,
    },
    frames: { bin_ms: 25, n_bins: 10, active_b64: Array(10).fill("") },
    brain: "malecns",
    sim_ms: 250,
    n_trials: 3,
    answered_at: "2026-09-18T00:00:00Z",
    lab: {
      trials: [],
      b0: 0,
      corrected_bias: 0.12,
      yes_side: "left",
      stimulated: { total: 0, left: 0, right: 0, by_modality: {} },
      active_neurons: 0,
      active_fraction: 0,
      total_spikes: 0,
      readout_latency_ms: null,
      top_cell_types: [],
      raster: { atlas_indices: [], cell_types: [], spikes_b64: "" },
      provenance: {
        brain: "malecns",
        n_neurons: 165122,
        n_connections: 6235682,
        n_synapses: 89731551,
        min_syn: 5,
        model: "shiu-lif",
        dt_ms: 0.1,
        sim_ms: 250,
        code_sha: "test",
        config_hash: "test",
        reproduce: "",
      },
    },
    ...overrides,
  };
}

describe("stateToAnimations", () => {
  it("always includes turn, with intensity from |lateral_bias|", () => {
    const triggers = stateToAnimations(makeAnswer({ answer: "no", lateral_bias: -0.3 }));
    const turn = triggers.find((t) => t.kind === "turn");
    expect(turn).toBeDefined();
    expect(turn?.intensity).toBeCloseTo(0.3);
  });

  it("turns toward the physical side that produced the answer, not a fixed yes=X convention", () => {
    for (const yesSide of ["left", "right"] as const) {
      for (const answer of ["yes", "no"] as const) {
        const triggers = stateToAnimations(makeAnswer({ answer, lateral_bias: -.01,
          lab: { ...makeAnswer().lab, yes_side: yesSide } }));
        expect(triggers[0]).toEqual({ kind: "turn", intensity: .01, answer, yesSide,
          toward: answer === "yes" ? yesSide : yesSide === "left" ? "right" : "left" });
        expect(triggers[1]).toEqual({ kind: "arousal", intensity: .4 });
      }
    }
  });

  it("skips readouts that are null (only shows real graph outputs)", () => {
    const triggers = stateToAnimations(makeAnswer());
    const kinds = triggers.map((t) => t.kind);
    expect(kinds).toContain("turn");
    expect(kinds).toContain("arousal");
    expect(kinds).not.toContain("appetite");
    expect(kinds).not.toContain("fear");
    expect(kinds).not.toContain("backoff");
    expect(kinds).not.toContain("courtship");
  });

  it("includes every non-null readout, e.g. all five plus turn on a fully active male answer", () => {
    const triggers = stateToAnimations(
      makeAnswer({
        states: { appetite: 0.9, fear: 0.1, backoff: 0.2, courtship: 0.5, arousal: 0.6 },
      }),
    );
    expect(triggers.map((t) => t.kind).sort()).toEqual(
      ["appetite", "arousal", "backoff", "courtship", "fear", "turn"].sort(),
    );
  });

  it("clamps intensity into [0, 1]", () => {
    const triggers = stateToAnimations(makeAnswer({ lateral_bias: 5, states: { appetite: 2, fear: null, backoff: null, courtship: null, arousal: -3 } }));
    for (const t of triggers) {
      expect(t.intensity).toBeGreaterThanOrEqual(0);
      expect(t.intensity).toBeLessThanOrEqual(1);
    }
  });
});
