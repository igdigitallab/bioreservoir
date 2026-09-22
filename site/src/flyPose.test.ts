import { describe, expect, it } from "vitest";
import { answerTurnAt, answerYaw, ANSWER_YAW, canSwitchFlurryPeriod, decisiveness, DECISIVE_BIAS, createFlyPose, DELIBERATE_FLURRY_PERIOD, DELIBERATE_SLOT_SECONDS, DELIBERATE_YAW, deliberateYaw, driveAt, envelope, FLURRY_PERIOD, flurryDuration, flurryProgress, flurryStart, gaitPhase, groomingAmount, HOVER_HEIGHT, hoverHeight, IDLE_SEED, poseAt, solveLeg, triggerDrive, ZERO_DRIVE } from "./flyPose";
import type { AnimationKind } from "./states";

describe("fly replay envelope", () => {
  it("preserves the original 20/60/20 attack, sustain and release", () => {
    for (const [p, value] of [[0, 0], [.1, .5], [.2, 1], [.5, 1], [.8, 1], [.9, .5], [1, 0]]) {
      expect(envelope(p)).toBeCloseTo(value);
    }
    for (const p of [-1, 2, NaN, Infinity]) expect(envelope(p)).toBe(0);
  });
  it("returns to neutral and replaces previous behaviours", () => {
    const target = triggerDrive([{ kind: "fear", intensity: .7 }]);
    expect(driveAt(target, .1).fear).toBeCloseTo(.35);
    expect(driveAt(target, 1)).toEqual(ZERO_DRIVE);
    expect(triggerDrive([])).toEqual(ZERO_DRIVE);
  });
  it("never introduces another behaviour", () => {
    for (const kind of Object.keys(ZERO_DRIVE) as AnimationKind[]) {
      const drive = triggerDrive([kind === "turn" ? { kind, intensity: .6, toward: "right", answer: "yes", yesSide: "right" } : { kind, intensity: .6 }]);
      for (const other of Object.keys(drive) as AnimationKind[]) {
        expect(drive[other]).toBe(other === kind ? (kind === "turn" ? 1 : .6) : 0);
      }
    }
  });
  it("clamps invalid strengths without mutating inputs", () => {
    expect(triggerDrive([{ kind: "fear", intensity: NaN }])).toEqual(ZERO_DRIVE);
    expect(triggerDrive([{ kind: "fear", intensity: 9 }]).fear).toBe(1);
    expect(triggerDrive([{ kind: "fear", intensity: -1 }]).fear).toBe(0);
  });
});

describe("anatomical poses", () => {
  it("mirrors physical turns without changing other behaviour poses", () => {
    const left = poseAt(triggerDrive([{ kind: "turn", intensity: 1, toward: "left", answer: "yes", yesSide: "left" }]), 0);
    const right = poseAt(triggerDrive([{ kind: "turn", intensity: 1, toward: "right", answer: "yes", yesSide: "right" }]), 0);
    expect(left.yaw).toBeCloseTo(ANSWER_YAW);
    expect(left.yaw).toBe(-right.yaw);
    expect(poseAt({ ...ZERO_DRIVE, fear: 1, courtship: 1 }, 1).yaw).toBeCloseTo(0);
  });
  it("extends only one wing for courtship and only the proboscis for appetite", () => {
    const rest = poseAt(ZERO_DRIVE, .4, false, { seconds: .4, playing: true, recovery: 1, seed: IDLE_SEED });
    const courtship = poseAt({ ...ZERO_DRIVE, courtship: 1 }, .4);
    expect(courtship.wings[0]).toBe(rest.wings[0]);
    expect(courtship.wings[1]).toBeGreaterThan(rest.wings[1] + .9);
    const appetite = poseAt({ ...ZERO_DRIVE, appetite: 1 }, .4);
    expect({ ...appetite, proboscis: 0 }).toEqual(rest);
  });
  it("alternates backward walking tripods", () => {
    const p = poseAt({ ...ZERO_DRIVE, backoff: 1 }, .2);
    expect(p.backward).toBeGreaterThan(0);
    expect(p.legs[0].sweep).toBeCloseTo(p.legs[2].sweep);
    expect(p.legs[0].sweep).toBeCloseTo(p.legs[4].sweep);
    expect(p.legs[0].sweep).toBeCloseTo(-p.legs[1].sweep);
  });
  it("has no periodic or idle movement with reduced motion", () => {
    const all = { turn: 1, appetite: 1, fear: 1, backoff: 1, courtship: 1, arousal: 1 };
    expect(poseAt(all, 0, true)).toEqual(poseAt(all, 90, true));
    expect(poseAt(ZERO_DRIVE, 0, true).legs.every(leg => leg.lift === 0 && leg.sweep === 0)).toBe(true);
  });
  it("is deterministic when seeking a screenshot or replay", () => {
    const drive = { ...ZERO_DRIVE, backoff: .8, arousal: .5 };
    expect(poseAt(drive, .75)).toEqual(poseAt(drive, .75));
  });
  it("preserves bone lengths and the resting knee when solving a lifted foot", () => {
    expect(solveLeg([-.7, .4, 0], [-.35, -.56, 0], [0, 0, 0]).knee[1]).toBeCloseTo(.4);
    const { knee, ankle } = solveLeg([-.7, .4, 0], [-.35, -.56, 0], [0, .16, .12]);
    expect(Math.hypot(...knee)).toBeCloseTo(Math.hypot(.7, .4));
    expect(Math.hypot(...ankle.map((value, i) => value - knee[i]))).toBeCloseTo(Math.hypot(.35, .56));
    expect(ankle[2]).toBeCloseTo(.12);
  });
  it("clamps unreachable IK targets without invalid rotations", () => {
    const solved = solveLeg([1, 0, 0], [1, 0, 0], [100, 0, 0]);
    expect(solved.knee.every(Number.isFinite)).toBe(true);
    expect(Math.hypot(...solved.ankle)).toBeLessThan(2);
  });
});


describe("decorative idle motion", () => {
  it("seeks a seeded schedule with starts 3–5 seconds apart and bursts of .6–1 second", () => {
    const starts = new Set<number>();
    for (let i = 0; i < 100; i++) {
      const start = flurryStart(i);
      const duration = flurryDuration(i);
      const interval = start - (i === 0 ? 0 : flurryStart(i - 1));
      expect(interval).toBeGreaterThanOrEqual(3);
      expect(interval).toBeLessThanOrEqual(5);
      expect(duration).toBeGreaterThanOrEqual(.6);
      expect(duration).toBeLessThanOrEqual(1);
      expect(flurryProgress(start - .00001)).toBe(-1);
      expect(flurryProgress(start)).toBeCloseTo(0);
      expect(flurryProgress(start + duration * .5)).toBeCloseTo(.5);
      expect(flurryProgress(start + duration + .00001)).toBe(-1);
      starts.add(interval);
    }
    expect(starts.size).toBeGreaterThan(90);
    expect(flurryStart(15, 17)).toBe(flurryStart(15, 17));
    expect(flurryStart(15, 17)).not.toBe(flurryStart(15, 18));
    for (const t of [-1, NaN, Infinity]) expect(flurryProgress(t)).toBe(-1);
  });
  it("lifts without crouching and lands smoothly with a bounded height", () => {
    expect(hoverHeight(.5)).toBeCloseTo(HOVER_HEIGHT);
    for (const p of [-1, 0, .12, .94, 1, NaN]) expect(hoverHeight(p)).toBe(0);
    for (let i = 0; i <= 100; i++) {
      expect(hoverHeight(i / 100)).toBeGreaterThanOrEqual(0);
      expect(hoverHeight(i / 100)).toBeLessThanOrEqual(HOVER_HEIGHT);
    }
    expect(hoverHeight(.12001)).toBeLessThan(1e-8);
    expect(hoverHeight(.93999)).toBeLessThan(1e-8);
    const start = flurryStart(0), duration = flurryDuration(0);
    const peak = poseAt(ZERO_DRIVE, start + duration * .5);
    expect(peak.legs.every(leg => leg.lift < 0)).toBe(true);
    expect(Math.max(...peak.wingLift)).toBeLessThan(.31);
    expect(peak.wingBlur).toBe(1);
    const landed = poseAt(ZERO_DRIVE, start + duration);
    expect(landed.height).toBe(0);
    expect(landed.wingBlur).toBeCloseTo(0);
  });
  it("uses alternating tripods, with only three feet lifting in each half-cycle", () => {
    const time = .6;
    expect(gaitPhase(time, 0)).toBe(gaitPhase(time, 2));
    expect(gaitPhase(time, 0)).toBe(gaitPhase(time, 4));
    expect(gaitPhase(time, 1) - gaitPhase(time, 0)).toBeCloseTo(Math.PI);
    const pose = poseAt(ZERO_DRIVE, time);
    expect(pose.legs.filter(leg => leg.lift > .001)).toHaveLength(3);
    expect(pose.legs[0].lift).toBeGreaterThan(.1);
    expect(pose.legs[1].lift).toBe(0);
  });
  it("occasionally brings just the front feet together to rub", () => {
    expect(groomingAmount(0)).toBe(0);
    expect(groomingAmount(2.4)).toBe(1);
    expect(groomingAmount(3.3)).toBe(0);
    expect(groomingAmount(11.4)).toBe(1);
    const pose = poseAt(ZERO_DRIVE, 2.4);
    expect(pose.legs[0].inward).toBeGreaterThan(.3);
    expect(pose.legs[3].inward).toBeLessThan(-.3);
    expect(pose.legs[0].curl).toBeCloseTo(.6);
    expect(pose.legs[3].curl).toBeCloseTo(-.6);
    for (const i of [1, 2, 4, 5]) expect(pose.legs[i].lift).toBe(0);
  });
  it("disables the flurry and all idle movement for reduced motion", () => {
    const peak = flurryStart(0) + flurryDuration(0) * .5;
    expect(flurryProgress(peak, IDLE_SEED, true)).toBe(-1);
    const rest = poseAt(ZERO_DRIVE, 0, true);
    for (const t of [.6, 2.4, peak, peak + .3, 100]) expect(poseAt(ZERO_DRIVE, t, true)).toEqual(rest);
    expect(rest.height).toBe(0);
    expect(rest.wingBlur).toBe(0);
  });
  it("suppresses decoration for the entire replay, including zero-strength endpoints", () => {
    const peak = flurryStart(0) + flurryDuration(0) * .5;
    const idle = { seconds: peak, playing: true, recovery: 1, seed: IDLE_SEED };
    expect(flurryProgress(peak, IDLE_SEED, false, true)).toBe(-1);
    const suppressed = poseAt(ZERO_DRIVE, peak, false, idle);
    expect(suppressed.height).toBeCloseTo(0);
    expect(suppressed.wingBlur).toBe(0);
    for (const leg of suppressed.legs) {
      expect(leg.lift).toBeCloseTo(0);
      expect(leg.sweep).toBeCloseTo(0);
      expect(leg.inward).toBeCloseTo(0);
      expect(leg.curl).toBeCloseTo(0);
    }
    const fear = { ...ZERO_DRIVE, fear: 1 };
    expect(poseAt(fear, .5, false, idle)).toEqual(poseAt(fear, .5, false, { ...idle, seconds: 0 }));
    idle.playing = false;
    idle.seconds = .6;
    idle.recovery = 0;
    expect(poseAt(ZERO_DRIVE, .6, false, idle).legs[0].lift).toBe(0);
    idle.recovery = .25;
    expect(poseAt(ZERO_DRIVE, .6, false, idle).legs[0].lift).toBeCloseTo(.0525);
    idle.recovery = .5;
    expect(poseAt(ZERO_DRIVE, .6, false, idle).legs[0].lift).toBeCloseTo(.105);
  });
  it("reuses live pose, drive and IK buffers without retaining old motion", () => {
    const pose = createFlyPose(), legs = pose.legs, wing = pose.wings;
    expect(poseAt(ZERO_DRIVE, 2.4, false, undefined, pose)).toBe(pose);
    expect(pose.legs).toBe(legs);
    expect(pose.wings).toBe(wing);
    poseAt(ZERO_DRIVE, 0, true, undefined, pose);
    expect(pose).toEqual(poseAt(ZERO_DRIVE, 0, true));
    const drive = { ...ZERO_DRIVE };
    expect(driveAt({ ...ZERO_DRIVE, fear: 1 }, .5, drive)).toBe(drive);
    driveAt(ZERO_DRIVE, 1, drive);
    expect(drive).toEqual(ZERO_DRIVE);
    const solved = { knee: [0, 0, 0], ankle: [0, 0, 0] } as Parameters<typeof solveLeg>[3];
    expect(solveLeg([1, 0, 0], [0, 1, 0], [0, 0, 0], solved)).toBe(solved);
  });
});

describe("answer turn choreography", () => {
  const turn = { kind: "turn", toward: "left", answer: "yes", yesSide: "left", intensity: .01 } as const;
  it("maps magnitude to a bounded display decisiveness, independently of sign", () => {
    expect(DECISIVE_BIAS).toBe(.03);
    for (const [bias, expected] of [[0, 0], [.002, 1 / 15], [.01, 1 / 3], [.015, .5], [.03, 1], [1, 1], [Infinity, 1], [NaN, 0]]) {
      expect(decisiveness(bias)).toBeCloseTo(expected);
      expect(decisiveness(-bias)).toBeCloseTo(expected);
    }
  });
  it("always finishes at 70 degrees toward the real side, even with zero bias", () => {
    for (const toward of ["left", "right"] as const) {
      for (const intensity of [0, .002, .01, .03, 1]) {
        for (const duration of [0, .1, 2.5, 6]) {
          const result = answerTurnAt({ ...turn, toward, intensity }, duration, duration);
          expect(result.yaw).toBeCloseTo(answerYaw(toward));
          expect(result.active).toBe(true);
        }
      }
    }
  });
  it("looks both ways for a weak bias, then commits, while strong bias snaps", () => {
    expect(answerTurnAt(turn, .15, 2.5).yaw).toBeGreaterThan(0);
    expect(answerTurnAt(turn, .52, 2.5).yaw).toBeLessThan(0);
    expect(answerTurnAt(turn, 1.5, 2.5).yaw).toBeGreaterThan(.9);
    expect(answerTurnAt({ ...turn, intensity: .03 }, .45, 2.5).yaw).toBeCloseTo(ANSWER_YAW);
    expect(answerTurnAt(turn, .45, 2.5).yaw).toBeLessThan(ANSWER_YAW * .2);
  });
  it("holds for six seconds after replay and eases out with the labels", () => {
    for (const seconds of [2.5, 5, 8.5]) {
      expect(answerTurnAt(turn, seconds, 2.5)).toEqual({ yaw: ANSWER_YAW, opacity: 1, active: true, chosen: 1 });
    }
    const release = answerTurnAt(turn, 8.9, 2.5);
    expect(release.yaw).toBeCloseTo(ANSWER_YAW / 2);
    expect(release.opacity).toBeCloseTo(.5);
    expect(answerTurnAt(turn, 9.3, 2.5)).toEqual({ yaw: 0, opacity: 0, active: false, chosen: 0 });
    expect(answerTurnAt(undefined, .5, 2.5).active).toBe(false);
  });
  it("starts an interrupted turn continuously and ends on the new side", () => {
    const fromYaw = answerTurnAt(turn, 3, 2.5).yaw;
    const next = { ...turn, toward: "right" as const };
    expect(answerTurnAt(next, 0, 2.5, false, fromYaw).yaw).toBe(fromYaw);
    expect(answerTurnAt(next, 2.5, 2.5, false, fromYaw).yaw).toBe(-ANSWER_YAW);
  });
  it("uses only a static final heading and label under reduced motion", () => {
    for (const t of [0, .1, 1, 2.5, 8.49]) {
      expect(answerTurnAt(turn, t, 2.5, true)).toEqual({ yaw: ANSWER_YAW, opacity: 1, active: true, chosen: 1 });
    }
    expect(answerTurnAt(turn, 8.5, 2.5, true)).toEqual({ yaw: 0, opacity: 0, active: false, chosen: 0 });
  });
  it("holds the answer for good when asked to (the reveal keeps its caption and verdict)", () => {
    for (const seconds of [2.5, 9.3, 60, 3600]) {
      expect(answerTurnAt(turn, seconds, 2.5, false, 0, Infinity)).toEqual({ yaw: ANSWER_YAW, opacity: 1, active: true, chosen: 1 });
    }
  });
  it("lights the chosen label only after the fly has committed to its heading", () => {
    for (const intensity of [0, .01, .03]) {
      const weak = { ...turn, intensity };
      // Mid-turn (well before the commit point) the labels still look alike.
      expect(answerTurnAt(weak, .05, 2.5).chosen).toBe(0);
      let committed = 0;
      for (let t = 0; t <= 2.5; t += .01) {
        const pose = answerTurnAt(weak, t, 2.5);
        if (pose.chosen > 0) { committed = t; break; }
      }
      // Lit only once the heading is final.
      expect(answerTurnAt(weak, committed, 2.5).yaw).toBeCloseTo(ANSWER_YAW, 2);
      expect(answerTurnAt(weak, 2.6, 2.5).chosen).toBe(1);
    }
  });
});

// The fly must look busy while a question is in the simulator, without any frame of that
// decoration being readable as an answer (design requirement, 2026-09-19).
describe("deliberation while a question is being simulated", () => {
  const samples = Array.from({ length: 400 }, (_, i) => deliberateYaw(i * .05));

  it("starts facing forward and never turns as far as an answer does", () => {
    expect(deliberateYaw(0)).toBe(0);
    for (const yaw of samples) expect(Math.abs(yaw)).toBeLessThanOrEqual(DELIBERATE_YAW + 1e-9);
    expect(DELIBERATE_YAW).toBeLessThan(ANSWER_YAW / 3);
  });

  it("keeps looking from side to side instead of settling on one", () => {
    const held = [1, 2, 3, 4, 5, 6].map((slot) => deliberateYaw((slot + .9) * DELIBERATE_SLOT_SECONDS));
    for (const yaw of held) expect(Math.abs(yaw)).toBeGreaterThan(DELIBERATE_YAW * .3);
    for (let i = 1; i < held.length; i++) expect(Math.sign(held[i]!)).toBe(-Math.sign(held[i - 1]!));
  });

  it("moves smoothly: no jump between consecutive frames", () => {
    for (let i = 1; i < samples.length; i++) {
      expect(Math.abs(samples[i]! - samples[i - 1]!)).toBeLessThan(DELIBERATE_YAW * .5);
    }
  });

  it("holds still for reduced motion and ignores nonsense clocks", () => {
    expect(deliberateYaw(3.2, IDLE_SEED, true)).toBe(0);
    for (const t of [-1, 0, NaN, Infinity]) expect(deliberateYaw(t)).toBe(0);
  });

  it("takes off more often while deciding than when nothing is asked", () => {
    expect(DELIBERATE_FLURRY_PERIOD).toBeLessThan(FLURRY_PERIOD);
    const lifts = (period: number) =>
      Array.from({ length: 300 }, (_, i) => flurryProgress(i * .1, IDLE_SEED, false, false, period)).filter((p) => p >= 0).length;
    expect(lifts(DELIBERATE_FLURRY_PERIOD)).toBeGreaterThan(lifts(FLURRY_PERIOD));
  });

  it("only changes cadence with the fly on the ground, and gets a chance to within a few seconds", () => {
    const periods = [FLURRY_PERIOD, DELIBERATE_FLURRY_PERIOD] as const;
    let switchable = 0;
    for (let i = 0; i < 2000; i++) {
      const seconds = i * .02;
      for (const [from, to] of [periods, [periods[1], periods[0]] as const]) {
        if (!canSwitchFlurryPeriod(seconds, IDLE_SEED, false, false, from, to)) continue;
        switchable++;
        // The height is the same (grounded) under both cadences, so no frame can jump.
        expect(hoverHeight(flurryProgress(seconds, IDLE_SEED, false, false, from))).toBe(0);
        expect(hoverHeight(flurryProgress(seconds, IDLE_SEED, false, false, to))).toBe(0);
      }
    }
    expect(switchable).toBeGreaterThan(0);
    // A chance to switch always comes soon: never more than a couple of seconds of hop overlap.
    const blocked = Array.from({ length: 400 }, (_, i) =>
      canSwitchFlurryPeriod(i * .02, IDLE_SEED, false, false, FLURRY_PERIOD, DELIBERATE_FLURRY_PERIOD));
    let longest = 0, run = 0;
    for (const ok of blocked) { run = ok ? 0 : run + 1; longest = Math.max(longest, run); }
    expect(longest * .02).toBeLessThan(2.5);
  });
});
