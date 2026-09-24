import { describe, expect, it } from "vitest";
import { pickNext, relativeAgo } from "./liveStage";
import type { Answer, AnswerSummary, ThinkingItem } from "../api/types";

const deciding = (id: string) => ({ kind: "deciding" as const, item: { id, question: `q${id}`, started_at: "", embargoed: false } as ThinkingItem });
const answer = (id: string, own = false) => ({ kind: "answer" as const, answer: { id } as Answer, label: "", own });
const sealed = (id: string) => ({ kind: "sealed" as const, summary: { id } as AnswerSummary, label: "", own: false });
const ids = (items: Array<{ kind: string } & Record<string, unknown>>) =>
  items.map((p) => `${p.kind}:${(p as { item?: { id: string } }).item?.id ?? (p as { answer?: { id: string } }).answer?.id ?? (p as { summary?: { id: string } }).summary?.id}`);

describe("live stage order", () => {
  it("shows an answer before the next question's 'deciding', in queue order", () => {
    const { next, rest } = pickNext([deciding("6"), answer("5")]);
    expect(ids([next!])).toEqual(["answer:5"]);
    expect(ids(rest)).toEqual(["deciding:6"]);
  });
  it("drops a 'deciding' once its answer is pending or a later question has started", () => {
    const { next, rest } = pickNext([deciding("5"), deciding("6"), answer("5")]);
    expect(ids([next!, ...rest])).toEqual(["answer:5", "deciding:6"]);
    expect(ids([pickNext([deciding("7"), sealed("7")]).next!])).toEqual(["sealed:7"]);
  });
  it("puts the visitor's own answer first", () => {
    const { next, rest } = pickNext([answer("3"), deciding("4"), answer("9", true)]);
    expect(ids([next!])).toEqual(["answer:9"]);
    expect(ids(rest)).toEqual(["answer:3", "deciding:4"]);
  });
  it("returns nothing for an empty stage queue", () => {
    expect(pickNext([]).next).toBeUndefined();
  });
});

describe("relative time", () => {
  const now = Date.parse("2026-09-19T12:00:00Z");
  it("reads naturally", () => {
    expect(relativeAgo("2026-09-19T11:59:30Z", now)).toBe("just now");
    expect(relativeAgo("2026-09-19T11:48:00Z", now)).toBe("12 min ago");
    expect(relativeAgo("2026-09-19T09:00:00Z", now)).toBe("3 h ago");
  });
});
