import { describe, expect, it } from "vitest";
import { formatPace, formatWait, livePosition, loadMyQuestions, saveMyQuestion, type MyQuestion } from "./queue";

function memoryStore() {
  const data = new Map<string, string>();
  return { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v) };
}

describe("live position", () => {
  it("moves up by one for every question the worker has taken since", () => {
    expect(livePosition(347, 1000, 1000)).toBe(347);
    expect(livePosition(347, 1000, 1010)).toBe(337);
  });
  it("never drops below first in line, even if the counter jumps past it", () => {
    expect(livePosition(3, 50, 60)).toBe(1);
    expect(livePosition(3, 50, 40)).toBe(3); // a stale counter never pushes it back
  });
});

describe("wait estimate", () => {
  it("is always approximate and coarser for longer waits", () => {
    expect(formatWait(1, 40)).toBe("about a minute");
    expect(formatWait(10, 60)).toBe("about 10 min");
    expect(formatWait(200, 60)).toBe("about 3 h 20 min");
    expect(formatWait(180, 60)).toBe("about 3 h");
    expect(formatWait(1200, 60)).toBe("about 20 h");
    expect(formatWait(5000, 60)).toBe("about 3 days");
  });
  it("falls back to one a minute when the pace is unknown", () => {
    expect(formatWait(30, NaN)).toBe("about 30 min");
    expect(formatWait(30, 0)).toBe("about 30 min");
  });
  it("words the pace", () => {
    expect(formatPace(60)).toBe("about one answer a minute");
    expect(formatPace(20)).toBe("about 3 answers a minute");
    expect(formatPace(150)).toBe("about one answer every 3 min");
  });
});

describe("my questions", () => {
  const q = (id: string, status: MyQuestion["status"] = "queued"): MyQuestion =>
    ({ id, question: `q${id}`, status, position: 5, claimedAt: 10, askedAt: "2026-09-19T00:00:00Z" });
  it("keeps the newest first and updates in place by id", () => {
    const store = memoryStore();
    saveMyQuestion(q("1"), store);
    saveMyQuestion(q("2"), store);
    saveMyQuestion(q("1", "answered"), store);
    const list = loadMyQuestions(store);
    expect(list.map((x) => x.id)).toEqual(["1", "2"]);
    expect(list[0]!.status).toBe("answered");
  });
  it("keeps only the last six", () => {
    const store = memoryStore();
    for (let i = 0; i < 9; i++) saveMyQuestion(q(String(i)), store);
    expect(loadMyQuestions(store)).toHaveLength(6);
  });
  it("survives garbage and missing storage", () => {
    const store = memoryStore();
    store.setItem("bioreservoir.myQuestions.v1", "{not json");
    expect(loadMyQuestions(store)).toEqual([]);
    expect(loadMyQuestions(undefined)).toEqual([]);
  });
});
