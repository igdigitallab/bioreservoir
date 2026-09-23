import { describe, expect, it } from "vitest";
import { DEFAULT_TAB, TAB_ORDER, loadLiked, mergeNewest, mergeRanked, saveLiked, sortTop, verdictKind, verdictLabel } from "./questions";
import type { AnswerSummary } from "../api/types";

const answer = (id: string, over: Partial<AnswerSummary> = {}): AnswerSummary => ({
  id,
  question: `Q${id}`,
  answer: "yes",
  embargoed: false,
  yes_side: "left",
  lateral_bias: 0.01,
  turn_strength: 0.1,
  answered_at: "2026-09-19T10:00:00+00:00",
  likes: 0,
  ...over,
});

describe("the questions archive", () => {
  it("puts a freshly answered question on top and never lists it twice", () => {
    const list = [answer("3"), answer("2")];
    const withNew = mergeNewest(list, answer("4"));
    expect(withNew.map((a) => a.id)).toEqual(["4", "3", "2"]);
    // The same answer arriving again (SSE replay, reconnect) replaces its row, it does not add one.
    const again = mergeNewest(withNew, answer("3", { likes: 5 }));
    expect(again.map((a) => a.id)).toEqual(["3", "4", "2"]);
    expect(again.filter((a) => a.id === "3")).toHaveLength(1);
    expect(again[0]!.likes).toBe(5);
  });

  it("opens on Most liked, with Latest one tap away", () => {
    expect(DEFAULT_TAB).toBe("top");
    expect(TAB_ORDER[0]).toBe(DEFAULT_TAB);
    expect([...TAB_ORDER].sort()).toEqual(["recent", "top"]);
  });

  it("ranks by likes, newest first among equals", () => {
    const ranked = sortTop([answer("1", { likes: 2 }), answer("5"), answer("3", { likes: 2 }), answer("4", { likes: 7 })]);
    expect(ranked.map((a) => a.id)).toEqual(["4", "3", "1", "5"]);
  });

  it("slots a live answer into Most liked under the liked rows, never twice", () => {
    const list = sortTop([answer("1", { likes: 3 }), answer("2")]);
    const withNew = mergeRanked(list, answer("6"));
    expect(withNew.map((a) => a.id)).toEqual(["1", "6", "2"]);
    const again = mergeRanked(withNew, answer("6", { likes: 4 }));
    expect(again.map((a) => a.id)).toEqual(["6", "1", "2"]);
    expect(again.filter((a) => a.id === "6")).toHaveLength(1);
  });

  it("colours the verdict chip by verdict, and a held-back one as held back", () => {
    expect(verdictKind(answer("1"))).toBe("yes");
    expect(verdictKind(answer("2", { answer: "no" }))).toBe("no");
    expect(verdictKind(answer("3", { embargoed: true, answer: "yes" }))).toBe("sealed");
    expect(verdictKind(answer("4", { answer: null }))).toBe("sealed");
  });

  it("keeps an embargoed verdict out of the list, like the stage does", () => {
    expect(verdictLabel(answer("1", { answer: "no" }))).not.toBe(verdictLabel(answer("2")));
    // An embargoed row can still carry a verdict (the asker's own copy of the summary): the
    // label must hide it, so this asserts on `embargoed` alone, not on a null answer.
    const held = verdictLabel(answer("3", { embargoed: true, answer: "yes" }));
    expect(held).toBe(verdictLabel(answer("4", { answer: null })));
    expect(held).not.toBe(verdictLabel(answer("5", { answer: "yes" })));
    expect(held.toLowerCase()).not.toContain("yes");
    expect(held.toLowerCase()).not.toContain("no");
  });

  it("remembers this browser's own likes, and survives junk in storage", () => {
    const data = new Map<string, string>();
    const store = {
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => void data.set(k, v),
    };
    expect(loadLiked(store).size).toBe(0);
    saveLiked(new Set(["7", "9"]), store);
    expect([...loadLiked(store)].sort()).toEqual(["7", "9"]);
    data.set("bioreservoir.liked.v1", "{not json");
    expect(loadLiked(store).size).toBe(0);
    expect(loadLiked(undefined).size).toBe(0);
  });
});
