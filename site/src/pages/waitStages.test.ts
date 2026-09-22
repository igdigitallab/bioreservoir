import { describe, expect, it } from "vitest";
import { stageLines } from "./waitStages";

describe("stageLines", () => {
  it("shows the queue position as the only active stage while queued", () => {
    const lines = stageLines({ kind: "queued", position: 3 });
    expect(lines[0]).toEqual({ label: "In the queue: #3", state: "active" });
    expect(lines.slice(1).every((l) => l.state === "todo")).toBe(true);
  });

  it("marks encoding active only during the first second of thinking", () => {
    expect(stageLines({ kind: "thinking", elapsedS: 0.4 })[1].state).toBe("active");
    expect(stageLines({ kind: "thinking", elapsedS: 2 })[1].state).toBe("done");
  });

  it("shows real elapsed seconds on the simulation stage, never a percentage", () => {
    const sim = stageLines({ kind: "thinking", elapsedS: 12.7 })[2];
    expect(sim.state).toBe("active");
    expect(sim.label).toContain("· 12 s");
    expect(sim.label).not.toContain("%");
  });
});
