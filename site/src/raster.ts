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

export interface RasterRow {
  /** Index into raster.cell_types / raster.atlas_indices — the neuron this display row is. */
  row: number;
  /** How many times this neuron fired in the trial. */
  count: number;
  /** Time of its first spike, ms. */
  firstMs: number;
}

export interface RasterLayout {
  /** Only neurons that actually fired, busiest first (ties broken by the neuron's own index —
   * NOT by spike time, which draws a tidy diagonal across the quiet rows that looks like a
   * finding and is only an artefact of the sort). */
  rows: RasterRow[];
  /** Pixel height of one row, chosen so the whole plot fits `maxPlotHeight`. */
  rowHeight: number;
  /** Neurons recorded for this answer (raster.cell_types.length). */
  totalRows: number;
  /** Recorded neurons that never fired in this trial. */
  silentRows: number;
  /** Active rows that did not fit under `maxPlotHeight` even at the minimum row height. */
  droppedRows: number;
  spikeCount: number;
}

const MIN_ROW_HEIGHT = 2;
const MAX_ROW_HEIGHT = 7;

/** Turn decoded spikes into what the canvas should actually draw.
 *
 * The raster records a fixed sample of neurons (300 today), but in a 250 ms trial only about a
 * quarter of them fire at all — drawing a row per recorded neuron produced a ~1800px tall canvas
 * that was three-quarters blank, and the browser then squeezed that tall image into the panel
 * width, which is what made the spikes look like stretched dots. So: silent neurons are not
 * drawn, the rest are sorted so the plot reads top-down from busiest to quietest, and the row
 * height is whatever makes the whole thing fit `maxPlotHeight`. The counts that got dropped are
 * returned, not swallowed — the caption says them out loud. */
export function buildRasterLayout(
  raster: LabRaster,
  spikes: RasterSpike[],
  opts: { maxPlotHeight: number },
): RasterLayout {
  const totalRows = raster.cell_types.length;
  const byRow = new Map<number, RasterRow>();
  for (const spike of spikes) {
    const existing = byRow.get(spike.row);
    if (existing) {
      existing.count += 1;
      existing.firstMs = Math.min(existing.firstMs, spike.timeMs);
    } else {
      byRow.set(spike.row, { row: spike.row, count: 1, firstMs: spike.timeMs });
    }
  }
  const active = [...byRow.values()].sort((a, b) => b.count - a.count || a.row - b.row);
  const maxRows = Math.max(1, Math.floor(opts.maxPlotHeight / MIN_ROW_HEIGHT));
  const rows = active.slice(0, maxRows);
  const rowHeight =
    rows.length === 0
      ? MAX_ROW_HEIGHT
      : Math.min(MAX_ROW_HEIGHT, Math.max(MIN_ROW_HEIGHT, Math.floor(opts.maxPlotHeight / rows.length)));
  return {
    rows,
    rowHeight,
    totalRows,
    silentRows: totalRows - active.length,
    droppedRows: active.length - rows.length,
    spikeCount: spikes.length,
  };
}

/** One honest line under the plot: what was drawn, and what was left out. Kept to three clauses
 * because it wraps to three lines on a phone otherwise — "77 of 300 fired" already says how many
 * stayed silent, so that number is not repeated. */
export function describeRaster(layout: RasterLayout): string {
  if (layout.spikeCount === 0) return `no spikes from ${layout.totalRows} recorded neurons in this trial`;
  const active = layout.rows.length + layout.droppedRows;
  const parts = [`${layout.spikeCount} spikes`, `${active} of ${layout.totalRows} recorded neurons fired`];
  parts.push(layout.droppedRows > 0 ? `${layout.rows.length} busiest drawn` : "silent rows not drawn");
  return parts.join(" · ");
}
