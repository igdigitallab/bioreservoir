// Behaviour strengths never come from the idle clock. Only states.ts triggers may drive a
// response; the clock supplies gait phase, not a fabricated neural readout.
import type { AnimationTrigger, FlySide, TurnTrigger } from "./states";

// A display scale for legible timing, not a confidence threshold or biological claim.
export const DECISIVE_BIAS = .03;
export const ANSWER_YAW = 70 * Math.PI / 180;
export const ANSWER_HOLD_SECONDS = 6;
export const ANSWER_RELEASE_SECONDS = .8;

/** Bias magnitude controls manner only; the API's physical side always owns direction. */
export function decisiveness(bias: number): number {
  return Number.isNaN(bias) ? 0 : Math.min(1, Math.abs(bias) / DECISIVE_BIAS);
}

export function answerYaw(side: FlySide): number {
  return (side === "left" ? 1 : -1) * ANSWER_YAW;
}

/** Seekable answer choreography. The little looks are decoration, never a new readout.
 * Hold from the replay deadline so even a snap leaves six seconds to read the answer; the reveal
 * passes `holdSeconds = Infinity`, because its caption and verdict stay on screen for good.
 * `chosen` (0..1) lights the answer label only once the fly has committed to its heading: lit
 * earlier, the label would announce the answer while the caption still says "deciding". */
export function answerTurnAt(turn: TurnTrigger | undefined, seconds: number, duration: number,
  reducedMotion = false, fromYaw = 0, holdSeconds = ANSWER_HOLD_SECONDS,
): { yaw: number; opacity: number; active: boolean; chosen: number } {
  const release = Math.max(0, duration) + holdSeconds;
  const end = release + (reducedMotion ? 0 : ANSWER_RELEASE_SECONDS);
  if (!turn || seconds < 0 || seconds >= end) return { yaw: 0, opacity: 0, active: false, chosen: 0 };
  const final = answerYaw(turn.toward);
  if (reducedMotion) return { yaw: final, opacity: 1, active: true, chosen: 1 };
  const d = decisiveness(turn.intensity);
  const commitTime = Math.max(0, duration) * (.92 - .74 * d);
  const p = commitTime > 0 ? Math.min(1, seconds / commitTime) : 1;
  const look = final * .18 * (1 - d);
  const hesitation = .48 * (1 - d);
  let yaw: number;
  if (hesitation > 0 && p < hesitation / 2) {
    yaw = fromYaw + (look - fromYaw) * smooth(p / (hesitation / 2));
  } else if (hesitation > 0 && p < hesitation) {
    yaw = look - 2 * look * smooth((p - hesitation / 2) / (hesitation / 2));
  } else {
    const start = hesitation > 0 ? -look : fromYaw;
    yaw = start + (final - start) * smooth((p - hesitation) / (1 - hesitation));
  }
  const fade = 1 - smooth((seconds - release) / ANSWER_RELEASE_SECONDS);
  const chosen = commitTime > 0 ? smooth((seconds - commitTime) / .25) : 1;
  return { yaw: yaw * fade, opacity: smooth(seconds / .15) * fade, active: true, chosen };
}

/** Decoration for the minutes a question spends in the simulator: the fly looks left, then
 * right, then left again on the spot. It is a clock animation like the hover and the grooming —
 * never a readout, and deliberately unlike an answer: a fifth of `ANSWER_YAW`, strictly
 * alternating and never settling, so no frame of a thinking fly can be screenshotted as a verdict
 * (2026-09-19: the fly should already be turning a little while it counts). */
export const DELIBERATE_YAW = 14 * Math.PI / 180;
export const DELIBERATE_SLOT_SECONDS = 1.6;

/** The heading the fly holds during slot `index`: sides strictly alternate, only the size varies. */
export function deliberateLook(index: number, seed = IDLE_SEED): number {
  return (index % 2 === 0 ? 1 : -1) * DELIBERATE_YAW * (.35 + .65 * randomAt(index, seed ^ 0x2f1d));
}

/** Saccade-and-fixate: the turn happens over the first 40% of a slot, then the fly holds it. */
export function deliberateYaw(seconds: number, seed = IDLE_SEED, reducedMotion = false): number {
  if (reducedMotion || !Number.isFinite(seconds) || seconds <= 0) return 0;
  const index = Math.floor(seconds / DELIBERATE_SLOT_SECONDS);
  const phase = seconds / DELIBERATE_SLOT_SECONDS - index;
  const from = index === 0 ? 0 : deliberateLook(index - 1, seed);
  return from + (deliberateLook(index, seed) - from) * smooth(phase / .4);
}

export interface FlyDrive {
  turn: number;
  appetite: number;
  fear: number;
  backoff: number;
  courtship: number;
  arousal: number;
}

export const ZERO_DRIVE: Readonly<FlyDrive> = Object.freeze({
  turn: 0, appetite: 0, fear: 0, backoff: 0, courtship: 0, arousal: 0,
});

export function envelope(progress: number): number {
  if (!Number.isFinite(progress) || progress <= 0 || progress >= 1) return 0;
  if (progress < .2) return progress / .2;
  if (progress > .8) return (1 - progress) / .2;
  return 1;
}

export function triggerDrive(triggers: readonly AnimationTrigger[]): FlyDrive {
  const drive = { ...ZERO_DRIVE };
  for (const trigger of triggers) {
    const strength = Number.isFinite(trigger.intensity) ? Math.max(0, Math.min(1, trigger.intensity)) : 0;
    if (trigger.kind === "turn") drive.turn = trigger.toward === "right" ? 1 : -1;
    else drive[trigger.kind] = strength;
  }
  return drive;
}

/** The optional output buffers keep the live animation allocation-free. */
export function driveAt(target: FlyDrive, progress: number, out: FlyDrive = { ...ZERO_DRIVE }): FlyDrive {
  const gain = envelope(progress);
  out.turn = target.turn * gain;
  out.appetite = target.appetite * gain;
  out.fear = target.fear * gain;
  out.backoff = target.backoff * gain;
  out.courtship = target.courtship * gain;
  out.arousal = target.arousal * gain;
  return out;
}

export const IDLE_SEED = 731;
export const HOVER_HEIGHT = .18; // About 4% of the model's body length.
/** Seconds between decorative take-offs. Shorter while a question is being simulated: a fly that
 * lifts off every few seconds reads as working, a still one reads as a frozen page. */
export const FLURRY_PERIOD = 4;
export const DELIBERATE_FLURRY_PERIOD = 3;
const TAU = Math.PI * 2;
const smooth = (value: number) => {
  const x = Math.max(0, Math.min(1, value));
  return x * x * (3 - 2 * x);
};
function randomAt(index: number, seed: number): number {
  let x = (seed ^ Math.imul(index + 1, 0x9e3779b9)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x21f0aaad);
  x = Math.imul(x ^ (x >>> 15), 0x735a2d97);
  return ((x ^ (x >>> 15)) >>> 0) / 4294967296;
}

/** Jittered `period`-second slots give period±0.5 s between starts, with O(1) random access. */
export function flurryStart(index: number, seed = IDLE_SEED, period = FLURRY_PERIOD): number {
  return (index + 1) * period + randomAt(index, seed) - .5;
}
export function flurryDuration(index: number, seed = IDLE_SEED): number {
  return .6 + .4 * randomAt(index, seed ^ 0x51ed);
}
/** -1 means grounded. Playback is explicit because its envelope is zero at both ends. */
export function flurryProgress(seconds: number, seed = IDLE_SEED, reducedMotion = false, playing = false,
  period = FLURRY_PERIOD): number {
  if (reducedMotion || playing || !Number.isFinite(seconds) || seconds < 0) return -1;
  const slot = Math.floor(seconds / period);
  for (let index = slot; index >= Math.max(0, slot - 1); index--) {
    const progress = (seconds - flurryStart(index, seed, period)) / flurryDuration(index, seed);
    if (progress >= 0 && progress < 1) return progress;
  }
  return -1;
}

/** May the take-off cadence change from `from` to `to` at this instant? Only while the fly is on
 * the ground under BOTH cadences: the slot boundaries move with the period, so switching mid-hop
 * teleports the specimen a full HOVER_HEIGHT up or down in one frame. */
export function canSwitchFlurryPeriod(seconds: number, seed: number, reducedMotion: boolean,
  playing: boolean, from: number, to: number): boolean {
  return flurryProgress(seconds, seed, reducedMotion, playing, from) < 0
    && flurryProgress(seconds, seed, reducedMotion, playing, to) < 0;
}

/** Wings lead the lift; a longer, eased descent ends with zero landing velocity. No crouch. */
export function hoverHeight(progress: number): number {
  if (progress <= .12 || progress >= .94 || !Number.isFinite(progress)) return 0;
  return HOVER_HEIGHT * smooth((progress - .12) / .3) * (1 - smooth((progress - .52) / .42));
}
/** L1/L3/R2 alternate with R1/R3/L2. One full small shuffle every 2.4 seconds. */
export function gaitPhase(seconds: number, leg: number): number {
  return seconds / 2.4 * TAU + ((leg % 3 + Math.floor(leg / 3)) % 2) * Math.PI;
}
export function groomingAmount(seconds: number): number {
  const phase = ((seconds % 9) + 9) % 9;
  return smooth((phase - 1.7) / .35) * (1 - smooth((phase - 2.8) / .4));
}

export interface IdlePlayback {
  seconds: number;
  playing: boolean;
  /** Seconds since replay finished; ease decoration back over half a second. */
  recovery: number;
  seed: number;
  /** Seconds between take-offs; `DELIBERATE_FLURRY_PERIOD` while a question is being simulated. */
  period?: number;
}
export interface LegPose { sweep: number; lift: number; inward: number; curl: number }
export interface FlyPose {
  yaw: number;
  backward: number;
  height: number;
  sway: number;
  proboscis: number;
  wings: [number, number];
  wingLift: [number, number];
  wingBlur: number;
  antennae: [number, number];
  legs: LegPose[];
}
export function createFlyPose(): FlyPose {
  return { yaw: 0, backward: 0, height: 0, sway: 0, proboscis: 0,
    wings: [0, 0], wingLift: [0, 0], wingBlur: 0, antennae: [0, 0],
    legs: Array.from({ length: 6 }, () => ({ sweep: 0, lift: 0, inward: 0, curl: 0 })) };
}

/** Idle never comes from brain readouts: shuffling, grooming and the tiny hover are
 * decoration only. Only states.ts triggers may supply behavioural drive strengths.
 * Flybody anatomical reference: https://doi.org/10.1038/s41586-025-09029-4 */
export function poseAt(drive: FlyDrive, seconds: number, reducedMotion = false,
  idle?: Readonly<IdlePlayback>, out = createFlyPose()): FlyPose {
  const time = reducedMotion ? 0 : seconds;
  const idleTime = reducedMotion ? 0 : idle?.seconds ?? seconds;
  const playing = (idle?.playing ?? false) || drive.turn !== 0 || drive.appetite > 0 || drive.fear > 0 ||
    drive.backoff > 0 || drive.courtship > 0 || drive.arousal > 0;
  const gain = reducedMotion || playing ? 0 : smooth((idle?.recovery ?? 1) / .5);
  const progress = flurryProgress(idleTime, idle?.seed, reducedMotion, playing, idle?.period);
  const buzz = progress < 0 ? 0 : smooth(progress / .12) * (1 - smooth((progress - .82) / .18)) * gain;
  const height = hoverHeight(progress) * gain;
  const airborne = height / HOVER_HEIGHT;
  const groom = groomingAmount(idleTime) * gain * (1 - buzz);
  const shuffle = gain * (1 - buzz) * (1 - groom);
  const jitter = reducedMotion ? 0 : drive.arousal;
  out.sway = shuffle * .025 * Math.sin(gaitPhase(idleTime, 0));
  for (let i = 0; i < 6; i++) {
    const phase = time * 4.2 + ((i % 3 + Math.floor(i / 3)) % 2) * Math.PI;
    const step = gaitPhase(idleTime, i);
    const swing = Math.max(0, Math.sin(step));
    const front = i % 3 === 0 ? groom : 0;
    const side = i < 3 ? 1 : -1;
    out.legs[i].sweep = drive.backoff * .12 * Math.sin(phase) + jitter * .035 * Math.sin(time * 31 + i * 2)
      + shuffle * .075 * Math.cos(step) + front * (-.16 + .055 * Math.sin(idleTime * 24 + side * .65))
      + airborne * .035 * (i % 3 - 1);
    out.legs[i].lift = drive.backoff * .16 * Math.max(0, Math.cos(phase))
      + shuffle * .105 * swing * swing + front * (.26 + .025 * Math.sin(idleTime * 24)) - airborne * .07;
    out.legs[i].inward = side * (front * .55 + airborne * .045) - out.sway;
    out.legs[i].curl = side * front * .6;
  }
  out.yaw = -drive.turn * ANSWER_YAW;
  out.backward = drive.backoff * (reducedMotion ? .3 : .13 * Math.min(Math.max(0, seconds), 4));
  out.height = height + drive.fear * (reducedMotion ? .035 : -.065 + .15 * Math.pow(Math.max(0, Math.sin(time * 7)), 4));
  out.proboscis = drive.appetite;
  for (let i = 0; i < 2; i++) {
    out.wings[i] = drive.fear * .55 + (i === 1 ? drive.courtship * (1.05 + (reducedMotion ? 0 : .065 * Math.sin(time * 45))) : 0)
      + gain * (1 - buzz) * (.026 * Math.sin(idleTime * 9 + i * .8) + .009 * Math.sin(idleTime * 17 + i))
      + buzz * (.24 + .075 * Math.sin(idleTime * TAU * 37));
    out.wingLift[i] = drive.fear * .85 + buzz * (.17 + .13 * Math.sin(idleTime * TAU * 37));
    out.antennae[i] = gain * .024 * Math.sin(idleTime * (i === 0 ? 2.8 : 2.5) + i)
      + jitter * .12 * Math.sin(time * (i === 0 ? 27 : 29) + i * 2);
  }
  out.wingBlur = buzz;
  return out;
}

export type Vec3 = [number, number, number];
export interface SolvedLeg { knee: Vec3; ankle: Vec3 }

/** Two-bone IK preserves measured lengths and uses the resting knee as the bend pole.
 * Scalar scratch values and a reusable result avoid garbage in the render loop. */
export function solveLeg(upper: Vec3, lower: Vec3, offset: Vec3,
  out: SolvedLeg = { knee: [0, 0, 0], ankle: [0, 0, 0] }): SolvedLeg {
  const a = Math.hypot(upper[0], upper[1], upper[2]), b = Math.hypot(lower[0], lower[1], lower[2]);
  let x = upper[0] + lower[0] + offset[0];
  let y = upper[1] + lower[1] + offset[1];
  let z = upper[2] + lower[2] + offset[2];
  const raw = Math.hypot(x, y, z);
  if (raw > 1e-8) { x /= raw; y /= raw; z /= raw; }
  else { x = 0; y = -1; z = 0; }
  const distance = Math.max(Math.abs(a - b) + 1e-6, Math.min(a + b - 1e-6, raw));
  const dot = upper[0] * x + upper[1] * y + upper[2] * z;
  let px = upper[0] - dot * x, py = upper[1] - dot * y, pz = upper[2] - dot * z;
  let poleLength = Math.hypot(px, py, pz);
  if (poleLength < 1e-8) {
    const bx = Math.abs(x) < .9 ? 1 : 0, by = bx === 1 ? 0 : 1;
    const projection = bx * x + by * y;
    px = bx - projection * x; py = by - projection * y; pz = -projection * z;
    poleLength = Math.hypot(px, py, pz);
  }
  const along = (a * a - b * b + distance * distance) / (2 * distance);
  const height = Math.sqrt(Math.max(0, a * a - along * along)) / poleLength;
  out.knee[0] = x * along + px * height;
  out.knee[1] = y * along + py * height;
  out.knee[2] = z * along + pz * height;
  out.ankle[0] = x * distance; out.ankle[1] = y * distance; out.ankle[2] = z * distance;
  return out;
}
