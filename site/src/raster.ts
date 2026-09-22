// Decodes the per-answer spike raster (Answer.lab.raster) and formats the decision math (L/R
// spike counts -> raw bias -> minus handedness b0 -> corrected bias -> yes/no side). Pure and
// framework-free (see raster.test.ts) — the canvas-drawing code that consumes this lives in
// pages/labReadout.ts.
import type { AnswerLab, LabRaster } from "./api/types";

export interface RasterSpike {
  /** Row index into raster.atlas_indices / raster.cell_types. */
  row: number;
  /** Spike time within the trial, in ms (0..sim_ms). */
  timeMs: number;
}

/** Decode trial-0 spikes: little-endian uint16 pairs (row, time in units of 0.1ms). */
export function decodeRasterSpikes(raster: LabRaster): RasterSpike[] {
  if (raster.spikes_b64.length === 0) return [];
  const binary = atob(raster.spikes_b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`raster.spikes_b64 is not a whole number of (row, time) uint16 pairs: ${bytes.byteLength} bytes`);
  }
  const view = new DataView(bytes.buffer);
  const n = bytes.byteLength / 4;
  const spikes: RasterSpike[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = view.getUint16(i * 4, true);
    const tenthMs = view.getUint16(i * 4 + 2, true);
    spikes[i] = { row, timeMs: tenthMs / 10 };
  }
  return spikes;
}

export interface DecisionMath {
  trialCount: number;
  /** Arithmetic mean of the real per-trial `bias` values (a direct average of provided numbers,
   * not a re-derivation from spike counts). */
  meanRawBias: number;
  b0: number;
  correctedBias: number;
  yesSide: "left" | "right";
}

/** Format the decision-math chain for display. All inputs are real fields from Answer.lab — this
 * function does no simulation and invents no numbers, only averages/labels what is given. */
export function buildDecisionMath(lab: AnswerLab): DecisionMath {
  const meanRawBias =
    lab.trials.length > 0 ? lab.trials.reduce((sum, t) => sum + t.bias, 0) / lab.trials.length : NaN;
  return {
    trialCount: lab.trials.length,
    meanRawBias,
    b0: lab.b0,
    correctedBias: lab.corrected_bias,
    yesSide: lab.yes_side,
  };
}
