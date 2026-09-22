// Decode the spike-replay payload the answer API sends: 10 bins of 25ms real simulated time
// each, base64-encoded little-endian uint32 arrays of atlas indices that fired in that bin.
// Pure, framework-free so it is unit-testable (see frames.test.ts) and reusable by both the
// home hero and the /stream layout.
import type { AnswerFrames } from "./api/types";

/** How much slower than real time the 10x25ms=250ms of real simulated spiking is replayed. */
export const REPLAY_STRETCH = 10;
/** Wall-clock duration of one replayed bin, in ms (25ms * REPLAY_STRETCH). */
export const BIN_DISPLAY_MS = 25 * REPLAY_STRETCH;
/** Total wall-clock replay duration (10 bins), in ms — shared by the brain flash and the fly
 * icon animation so both settle back to idle at the same moment. */
export const TOTAL_REPLAY_MS = BIN_DISPLAY_MS * 10;

/** Decode one bin's base64 payload into the atlas indices that spiked. */
export function decodeActiveBin(base64: string): Uint32Array {
  if (base64.length === 0) return new Uint32Array(0);
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`active_b64 bin is not a whole number of uint32s: ${bytes.byteLength} bytes`);
  }
  // Little-endian per the API contract; DataView reads explicitly so this does not depend on
  // host endianness (typed-array views would instead follow the platform's native order).
  const view = new DataView(bytes.buffer);
  const out = new Uint32Array(bytes.byteLength / 4);
  for (let i = 0; i < out.length; i++) {
    out[i] = view.getUint32(i * 4, true);
  }
  return out;
}

/** Decode every bin of a frames payload, in bin order. */
export function decodeFrames(frames: AnswerFrames): Uint32Array[] {
  if (frames.active_b64.length !== frames.n_bins) {
    throw new Error(
      `frames.active_b64 has ${frames.active_b64.length} entries, expected n_bins=${frames.n_bins}`,
    );
  }
  return frames.active_b64.map(decodeActiveBin);
}

export interface ReplayStep {
  binIndex: number;
  indices: Uint32Array;
  /** ms from the start of the replay this bin should start showing at. */
  atMs: number;
}

/** Flatten decoded bins into a replay schedule (10 bins -> ~2.5s at REPLAY_STRETCH=10). */
export function buildReplaySchedule(bins: Uint32Array[]): ReplayStep[] {
  return bins.map((indices, binIndex) => ({
    binIndex,
    indices,
    atMs: binIndex * BIN_DISPLAY_MS,
  }));
}
