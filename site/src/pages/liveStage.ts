// The live stage: what everyone on the page watches while the queue works (2026-09-19:
// every visitor should see the fly answering other people's questions as soon as the page loads).
// On load it shows `/api/now` (the question being simulated, or an honestly labelled replay of the
// last answer); then SSE drives it: `thinking` → the new question and its elapsed time, `answered`
// → the answer told in order. An answer stays up for a minimum dwell before the next question
// replaces it, so it can be read. The visitor's own answer jumps the line and stays longer.
//
// Nothing here is staged: every question, spike replay and verdict is a real run from the worker;
// the only non-live item is the replay on load, and its label says so.
import { h } from "../dom";
import { copy } from "../content";
import { getAnswer, getNow } from "../api/client";
import { fmtTime } from "../format";
import { formatPace } from "../queue";
import { createStageView, answerSchedule, type StageView } from "./stageView";
import type { Answer, AnswerSummary, QueueTick, ServerEvent, ThinkingItem } from "../api/types";

/** How long an answer stays on stage after its verdict before the next item may replace it. */
export const ANSWER_DWELL_MS = 9000;
export const OWN_ANSWER_DWELL_MS = 30000;

type Item =
  | { kind: "deciding"; item: ThinkingItem }
  | { kind: "answer"; answer: Answer; label: string; own: boolean }
  | { kind: "sealed"; summary: AnswerSummary; label: string; own: boolean };

function itemId(item: Item): number {
  return Number(item.kind === "deciding" ? item.item.id : item.kind === "answer" ? item.answer.id : item.summary.id);
}

/** Which item goes on stage next: pure, so the priority rules are testable.
 *  - the visitor's own answer first;
 *  - otherwise in queue order (ids are FIFO), an answer before the next question's "deciding";
 *  - a "deciding" item is dropped once it is stale (its answer is already pending, or the worker
 *    has started a later question): showing a stale "deciding" would be a lie. */
export function pickNext(pending: Item[]): { next: Item | undefined; rest: Item[] } {
  const ownIndex = pending.findIndex((p) => p.kind !== "deciding" && p.own);
  if (ownIndex >= 0) {
    const rest = pending.slice();
    const [next] = rest.splice(ownIndex, 1);
    return { next, rest };
  }
  const sorted = pending.slice().sort((a, b) => itemId(a) - itemId(b) || (a.kind === "deciding" ? -1 : 1));
  const decidingIds = sorted.filter((p) => p.kind === "deciding").map(itemId);
  const latestDeciding = decidingIds.length ? Math.max(...decidingIds) : -1;
  const live = sorted.filter(
    (p) => p.kind !== "deciding" || (itemId(p) === latestDeciding && !sorted.some((q) => q.kind !== "deciding" && itemId(q) === itemId(p))),
  );
  const [next, ...rest] = live;
  return { next, rest };
}

export function relativeAgo(iso: string, now = Date.now()): string {
  const s = Math.max(0, (now - Date.parse(iso)) / 1000);
  if (!Number.isFinite(s) || s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return fmtTime(iso);
}

export interface LiveStage {
  /** The "● LIVE · N waiting" line: the page places it across the whole stage panel. */
  statusEl: HTMLElement;
  /** The fly, the question and the verdict. */
  el: HTMLElement;
  /** The wide "Lab readout" panel — the page puts it across the whole stage, not in the
   * narrow fly column (see StageView.readoutEl). */
  readoutEl: HTMLElement;
  /** SSE events: the page forwards its single stream here. */
  handle: (event: ServerEvent) => void;
  /** Put this visitor's own answer on stage now (own answers jump the line). Election questions
   * included: the embargo keeps them off OTHER people's screens, not off the asker's. */
  playOwn: (answer: Answer, label: string) => void;
  dispose: () => void;
}

/** A fetch of a full answer is in flight: hold "deciding" items back meanwhile (the answer belongs
 * before them), but never longer than this. */
const FETCH_HOLD_MS = 4000;
const FETCH_JITTER_MS = 1200;

export function mountLiveStage(opts: {
  /** Brain replay for the answer on stage, started with the fly's turn. */
  onFlyBeat: (answer: Answer) => void;
  /** An answer's verdict is on screen (or it was skipped): the feed row may appear now. */
  onShown: (id: string) => void;
  /** Is this id one of the visitor's own questions? */
  isMine: (id: string) => boolean;
  onQueue?: (tick: QueueTick) => void;
}): LiveStage {
  const view: StageView = createStageView();
  const liveDot = h("span", { class: "live-dot", "aria-hidden": "true" });
  const statusText = h("span", {}, copy.stage.label_live);
  const status = h("p", { class: "stage-status label-mono" }, [liveDot, statusText]);
  const el = h("div", { class: "live-stage" }, [view.el]);

  let pending: Item[] = [];
  let busyUntil = 0;
  let wakeTimer: number | undefined;
  let disposed = false;
  let lastTick: QueueTick | undefined;
  /** The question the worker is on now and not yet answered: the stage returns to it whenever
   * nothing else is due, so it never sits on an old answer while a new question is running. */
  let latestThinking: ThinkingItem | undefined;
  let showing: { kind: Item["kind"] | "idle"; id?: string } = { kind: "idle" };
  let fetchesInFlight = 0;
  let fetchHoldUntil = 0;
  const answeredIds = new Set<string>();

  function renderStatus() {
    if (!lastTick) return;
    const n = lastTick.queue_length;
    const waiting = n === 0 ? copy.stage.status_empty : copy.stage.status_waiting.replace("{n}", n.toLocaleString("en-US"));
    const pace = formatPace(lastTick.avg_cycle_s);
    statusText.textContent = `${copy.stage.label_live} · ${waiting}${pace && n > 0 ? ` · ${pace}` : ""}`;
  }

  function wakeAt(ms: number) {
    if (wakeTimer !== undefined) window.clearTimeout(wakeTimer);
    wakeTimer = window.setTimeout(schedule, Math.max(0, ms - Date.now()));
  }

  function schedule() {
    if (disposed) return;
    if (wakeTimer !== undefined) window.clearTimeout(wakeTimer);
    wakeTimer = undefined;
    const now = Date.now();
    // A "deciding" item whose question is already answered, or older than the one running now, is stale.
    pending = pending.filter(
      (p) => p.kind !== "deciding" || (!answeredIds.has(p.item.id) && (!latestThinking || Number(p.item.id) >= Number(latestThinking.id))),
    );
    const ownWaiting = pending.some((p) => p.kind !== "deciding" && p.own);
    if (busyUntil > now && !ownWaiting) return wakeAt(busyUntil);
    const { next, rest } = pickNext(pending);
    if (next?.kind === "deciding" && fetchesInFlight > 0 && fetchHoldUntil > now) return wakeAt(fetchHoldUntil);
    pending = rest;
    if (next) {
      show(next);
      if (pending.length) schedule();
      return;
    }
    // Nothing queued for the stage: go back to the question being simulated right now.
    if (latestThinking && !(showing.kind === "deciding" && showing.id === latestThinking.id)) {
      show({ kind: "deciding", item: latestThinking });
    }
  }

  function show(item: Item) {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (item.kind === "deciding") {
      showing = { kind: "deciding", id: item.item.id };
      const labelText = opts.isMine(item.item.id) ? copy.stage.label_yours_deciding : copy.stage.label_deciding;
      view.showDeciding(item.item.question, item.item.started_at, labelText, item.item.yes_side);
      busyUntil = 0;
      return;
    }
    if (item.kind === "sealed") {
      showing = { kind: "sealed", id: item.summary.id };
      view.showSealed(item.summary.question, item.label);
      opts.onShown(item.summary.id);
      busyUntil = Date.now() + ANSWER_DWELL_MS;
      wakeAt(busyUntil);
      return;
    }
    const answer = item.answer;
    showing = { kind: "answer", id: answer.id };
    view.showAnswer(answer, {
      label: item.label,
      onFlyBeat: () => opts.onFlyBeat(answer),
      onVerdict: () => opts.onShown(answer.id),
    });
    busyUntil = Date.now() + answerSchedule(reducedMotion).verdict + (item.own ? OWN_ANSWER_DWELL_MS : ANSWER_DWELL_MS);
    wakeAt(busyUntil);
  }

  function enqueue(item: Item) {
    pending.push(item);
    // Keep the backlog short: under a burst only the latest few items are worth showing.
    if (pending.length > 4) {
      const dropped = pending.splice(0, pending.length - 4);
      for (const d of dropped) if (d.kind !== "deciding") opts.onShown(d.kind === "answer" ? d.answer.id : d.summary.id);
    }
    schedule();
  }

  /** Full answer (spike replay for the brain) from the CDN-cached, immutable answer route. Each
   * tab waits a random moment first: thousands of tabs asking for the same answer in the same
   * second would all miss the edge cache at once; spread over ~1 s, the first fills it. */
  function fetchAndEnqueue(summary: AnswerSummary, label: string) {
    fetchesInFlight += 1;
    fetchHoldUntil = Date.now() + FETCH_HOLD_MS;
    new Promise((resolve) => window.setTimeout(resolve, Math.random() * FETCH_JITTER_MS))
      .then(() => getAnswer(summary.id))
      .then((answer) => {
        if (!disposed) enqueue({ kind: "answer", answer, label, own: false });
      })
      .catch(() => opts.onShown(summary.id))
      .finally(() => {
        fetchesInFlight = Math.max(0, fetchesInFlight - 1);
        schedule();
      });
  }

  function onAnswered(summary: AnswerSummary) {
    answeredIds.add(summary.id);
    if (latestThinking?.id === summary.id) latestThinking = undefined;
    // The page fetches the visitor's own answer itself and calls playOwn (full answer, embargoed
    // or not); here only other people's answers.
    if (opts.isMine(summary.id)) return;
    if (summary.embargoed) enqueue({ kind: "sealed", summary, label: copy.stage.label_visitor, own: false });
    else fetchAndEnqueue(summary, copy.stage.label_visitor);
  }

  function loadNow() {
    getNow()
      .then((now) => {
        if (disposed) return;
        lastTick = now;
        renderStatus();
        opts.onQueue?.(now);
        if (now.thinking) {
          latestThinking = now.thinking;
          schedule();
        } else if (now.last && showing.kind === "idle") {
          const last = now.last;
          // On load this is a REPLAY of the last answer, not a live one: the label says so
          // (sealed items included — they used to claim "just now", 2026-09-19 prod check).
          const replayLabel = copy.stage.label_replay.replace("{ago}", relativeAgo(last.answered_at));
          if (last.embargoed) enqueue({ kind: "sealed", summary: last, label: replayLabel, own: false });
          else fetchAndEnqueue(last, replayLabel);
        } else if (!now.last && showing.kind === "idle") {
          view.showIdle(copy.stage.idle_empty);
        }
      })
      .catch((err) => {
        console.error("now fetch failed", err);
        if (showing.kind === "idle") view.showIdle(copy.stage.idle_offline);
      });
  }

  view.showIdle(copy.stage.loading);
  loadNow();

  return {
    statusEl: status,
    el,
    readoutEl: view.readoutEl,
    handle(event) {
      if (event.type === "thinking") {
        latestThinking = event.data;
        enqueue({ kind: "deciding", item: event.data });
      } else if (event.type === "answered") {
        onAnswered(event.data);
      } else if (event.type === "queue") {
        lastTick = event.data;
        renderStatus();
        opts.onQueue?.(event.data);
      } else if (event.type === "reconnected") {
        loadNow();
      }
    },
    playOwn(answer, label) {
      answeredIds.add(answer.id);
      if (latestThinking?.id === answer.id) latestThinking = undefined;
      enqueue({ kind: "answer", answer, label, own: true });
    },
    dispose() {
      disposed = true;
      if (wakeTimer !== undefined) window.clearTimeout(wakeTimer);
      view.dispose();
    },
  };
}
