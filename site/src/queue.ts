// The visitor's side of a long queue (2026-09-19: "a question may wait hours or days,
// that's fine"). Pure functions plus a small localStorage record, so a tab can be closed and the
// question still found again: the server only knows ids, this browser remembers which are its own.

/** Place in line now. The worker takes questions strictly in order, so every question it has
 * taken since ours was queued moved us up by one. `claimed_total` is the server's monotonic count
 * of taken questions (`/api/ask`, `/api/now`, SSE `queue`). */
export function livePosition(positionAtAsk: number, claimedAtAsk: number, claimedNow: number): number {
  const moved = Math.max(0, claimedNow - claimedAtAsk);
  return Math.max(1, positionAtAsk - moved);
}

/** "about 3 h 20 min" at the current pace. Always "about": the pace is a median of recent
 * answers, not a promise. Coarser as the wait grows, so the number never looks more exact than
 * it is. */
export function formatWait(position: number, avgCycleS: number): string {
  const cycle = Number.isFinite(avgCycleS) && avgCycleS > 0 ? avgCycleS : 60;
  const minutes = (Math.max(1, position) * cycle) / 60;
  if (minutes < 1.5) return "about a minute";
  if (minutes < 60) return `about ${Math.round(minutes)} min`;
  const hours = minutes / 60;
  if (hours < 10) {
    const whole = Math.floor(hours);
    const rest = Math.round((minutes - whole * 60) / 10) * 10;
    if (rest === 60) return `about ${whole + 1} h`;
    return rest === 0 ? `about ${whole} h` : `about ${whole} h ${rest} min`;
  }
  if (hours < 48) return `about ${Math.round(hours)} h`;
  return `about ${Math.round(hours / 24)} days`;
}

/** "about one answer a minute" / "about one answer every 2 min" — the pace, in words. */
export function formatPace(avgCycleS: number): string {
  if (!Number.isFinite(avgCycleS) || avgCycleS <= 0) return "";
  if (avgCycleS < 45) return `about ${Math.round(60 / avgCycleS)} answers a minute`;
  if (avgCycleS < 90) return "about one answer a minute";
  return `about one answer every ${Math.round(avgCycleS / 60)} min`;
}

export type MyQuestionStatus = "queued" | "thinking" | "answered" | "rejected";

export interface MyQuestion {
  id: string;
  question: string;
  status: MyQuestionStatus;
  /** Place in line and the server's taken-count when this browser last learned it. */
  position: number;
  claimedAt: number;
  askedAt: string;
  /** Answered before this visitor asked: same words, same answer. */
  repeat?: boolean;
  /** Shared a place in line with someone who asked the same words first. */
  joined?: boolean;
  answer?: "yes" | "no" | null;
  /** Why moderation turned it down (a question can be accepted into line and moderated later). */
  rejectedMessage?: string;
}

import { localStore, type KeyValueStore } from "./storage";

const STORAGE_NAME = "bioreservoir.myQuestions.v1";
const KEEP = 6;

export function loadMyQuestions(store: KeyValueStore | undefined = localStore()): MyQuestion[] {
  if (!store) return [];
  try {
    const parsed: unknown = JSON.parse(store.getItem(STORAGE_NAME) ?? "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((q): q is MyQuestion => typeof q?.id === "string" && typeof q?.question === "string");
  } catch {
    return [];
  }
}

/** Insert or update by id, newest first, keeping the last few. */
export function saveMyQuestion(q: MyQuestion, store: KeyValueStore | undefined = localStore()): MyQuestion[] {
  const next = [q, ...loadMyQuestions(store).filter((other) => other.id !== q.id)].slice(0, KEEP);
  try {
    store?.setItem(STORAGE_NAME, JSON.stringify(next));
  } catch {
    // quota or blocked storage: the question still works for this tab, it just won't be remembered
  }
  return next;
}
