import { describe, expect, it } from "vitest";
import { Vector3 } from "three";
import { answerYaw } from "./flyPose";
import { createFlyCamera, projectAnswerPoint } from "./flyScene";

describe("answer label projection", () => {
  it("places both labels beyond the head on its projected forward axis", () => {
    const camera = createFlyCamera();
    const head = new Vector3(0, -.0305, -.567);
    for (const side of ["left", "right"] as const) {
      const yaw = answerYaw(side);
      const axis = new Vector3(0, 1, 0);
      const projectedHead = head.clone().applyAxisAngle(axis, yaw).project(camera);
      const ahead = head.clone().add(new Vector3(0, 0, -.1)).applyAxisAngle(axis, yaw).project(camera);
      const point = projectAnswerPoint(camera, head, side);
      const dx = point.x * 2 - 1 - projectedHead.x;
      const dy = 1 - point.y * 2 - projectedHead.y;
      expect(dx * (ahead.x - projectedHead.x) + dy * (ahead.y - projectedHead.y)).toBeGreaterThan(0);
      expect(dx * (ahead.y - projectedHead.y) - dy * (ahead.x - projectedHead.x)).toBeCloseTo(0);
      expect(point.x).toBeGreaterThan(.1);
      expect(point.x).toBeLessThan(.9);
      expect(point.y).toBeGreaterThan(.1);
      expect(point.y).toBeLessThan(.9);
    }
    // A regression to CSS left/right would put both labels on the wrong side.
    expect(projectAnswerPoint(camera, head, "left").x).toBeGreaterThan(.5);
    expect(projectAnswerPoint(camera, head, "right").x).toBeLessThan(.5);
  });
});
