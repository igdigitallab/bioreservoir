// Maps an Answer's behavioural readouts to fly-animation triggers. Pure and framework-free
// (see states.test.ts). Per the build brief: "Animate a state only if its value is non-null" —
// a null readout (the calibration check found it does not respond to text input, see
// experiments/001-fly-oracle) must never appear as a fabricated animation.
import type { Answer } from "./api/types";

export type AnimationKind = "turn" | "appetite" | "fear" | "backoff" | "courtship" | "arousal";

export type FlySide = Answer["lab"]["yes_side"];

export interface TurnTrigger {
  kind: "turn";
  /** Normalized 0..1 drive strength for this animation. */
  intensity: number;
  /** Only present for "turn": the physical side (not an abstract yes/no) the fly ends up facing
   * — computed from answer.lab.yes_side (the per-question side->answer mapping, see
   * experiments/001-fly-oracle/README.md's "Handedness and side mapping"), so the visual turn
   * always matches which side actually produced this answer, never a hardcoded yes=right/no=left
   * convention. */
  toward: FlySide;
  answer: Answer["answer"];
  yesSide: FlySide;
}

export type AnimationTrigger = TurnTrigger | {
  kind: Exclude<AnimationKind, "turn">;
  intensity: number;
};

function clamp01(x: number): number {
  if (Number.isNaN(x)) return 0;
  return Math.min(1, Math.max(0, x));
}

/**
 * Build the ordered list of animations a fly should play for one answer. Only readouts with a
 * non-null value produce a trigger; "turn" is always included because `answer`/`lateral_bias`
 * are non-nullable on Answer.
 */
export function stateToAnimations(answer: Answer): AnimationTrigger[] {
  const yesSide = answer.lab.yes_side;
  const noSide = yesSide === "left" ? "right" : "left";
  const physicalSide = answer.answer === "yes" ? yesSide : noSide;

  const triggers: AnimationTrigger[] = [
    {
      kind: "turn",
      intensity: clamp01(Math.abs(answer.lateral_bias)),
      toward: physicalSide,
      answer: answer.answer,
      yesSide,
    },
  ];

  const { states } = answer;
  if (states.appetite !== null) {
    triggers.push({ kind: "appetite", intensity: clamp01(states.appetite) });
  }
  if (states.fear !== null) {
    triggers.push({ kind: "fear", intensity: clamp01(states.fear) });
  }
  if (states.backoff !== null) {
    triggers.push({ kind: "backoff", intensity: clamp01(states.backoff) });
  }
  if (states.courtship !== null) {
    triggers.push({ kind: "courtship", intensity: clamp01(states.courtship) });
  }
  if (states.arousal !== null) {
    triggers.push({ kind: "arousal", intensity: clamp01(states.arousal) });
  }
  return triggers;
}
