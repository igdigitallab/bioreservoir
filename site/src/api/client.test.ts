import { describe, expect, it } from "vitest";
import { unwrapAnswerLookup } from "./client";
import type { Answer } from "./types";

describe("answer lookup", () => {
  it("returns the Answer inside an answered row (the shape api.py get_answer serves)", () => {
    const answer = { id: "11", question: "Will Trump fly?", answer: "yes" } as Answer;
    expect(unwrapAnswerLookup({ id: 11, question: "Will Trump fly?", status: "answered", answer })).toBe(answer);
  });
  it("refuses rows that carry no answer yet", () => {
    expect(() => unwrapAnswerLookup({ id: 12, question: "q", status: "queued", position: 3 })).toThrow(/queued/);
    expect(() => unwrapAnswerLookup({ id: 13, question: "q", status: "rejected" })).toThrow(/rejected/);
  });
});
