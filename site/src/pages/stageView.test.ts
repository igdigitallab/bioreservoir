import { describe, expect, it } from "vitest";
import { answerSchedule, formatElapsed } from "./stageView";
import { TOTAL_REPLAY_MS } from "../frames";

describe("answer told in order", () => {
  it("shows the question, then the fly's turn, then the verdict once the turn is over", () => {
    const { fly, verdict } = answerSchedule(false);
    expect(fly).toBeGreaterThan(0);
    expect(verdict).toBeGreaterThanOrEqual(fly + TOTAL_REPLAY_MS);
  });
  it("shows everything at once under reduced motion", () => {
    expect(answerSchedule(true)).toEqual({ fly: 0, verdict: 0 });
  });
  it("counts the simulation time on the deciding caption", () => {
    expect(formatElapsed(12.7)).toBe("12 s");
    expect(formatElapsed(65)).toBe("1 min 5 s");
    expect(formatElapsed(-3)).toBe("0 s");
  });
});
