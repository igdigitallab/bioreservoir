// "Your questions": the visitor's own questions, however long the line is (2026-09-19: "they
// queue for 3, 4, 5 hours, a day, several days, that's fine"). Each one shows its live place
// in line and an honest wait at the current pace, or that the fly is deciding it, or its answer.
// Every question has a link that plays its answer once it exists, so the tab can be closed. The
// list lives in localStorage (queue.ts): the server only knows ids.
import { h, clear } from "../dom";
import { copy } from "../content";
import { shareUrl, getAnswerLookup } from "../api/client";
import { formatWait, livePosition, loadMyQuestions, saveMyQuestion, type MyQuestion } from "../queue";
import type { QueueTick } from "../api/types";

export interface MyQuestions {
  el: HTMLElement;
  isMine: (id: string) => boolean;
  add: (q: MyQuestion) => void;
  update: (id: string, patch: Partial<MyQuestion>) => void;
  onQueue: (tick: QueueTick) => void;
  /** Re-read where the waiting questions stand (e.g. after the event stream reconnects). */
  refresh: () => void;
  dispose: () => void;
}

const REFRESH_MS = 180_000;

export function mountMyQuestions(opts: {
  /** "Watch it again": the page puts this answered question back on the stage. */
  onWatch: (id: string) => void;
  /** An answered question found on load (answered while the tab was closed). */
  onFoundAnswered: (id: string) => void;
}): MyQuestions {
  const list = h("ul", { class: "my-questions-list" });
  const el = h("div", { class: "my-questions", hidden: "" }, [h("h3", { class: "label-mono" }, copy.mine.title), list]);
  let items: MyQuestion[] = loadMyQuestions();
  let tick: QueueTick | undefined;
  let disposed = false;

  function statusLine(q: MyQuestion): string {
    if (q.status === "rejected") return `${copy.mine.rejected}${q.rejectedMessage ? `: ${q.rejectedMessage}` : "."}`;
    if (q.status === "answered") {
      if (q.answer === undefined || q.answer === null) return q.repeat ? copy.mine.repeat : copy.mine.answered.replace("{answer}", "…");
      const word = q.answer === "yes" ? copy.answer.yes_label : copy.answer.no_label;
      return q.repeat ? `${copy.mine.answered.replace("{answer}", word)} · ${copy.mine.repeat}` : copy.mine.answered.replace("{answer}", word);
    }
    if (q.status === "thinking") return copy.mine.thinking;
    const pos = tick ? livePosition(q.position, q.claimedAt, tick.claimed_total) : q.position;
    const wait = formatWait(pos, tick?.avg_cycle_s ?? NaN);
    return copy.mine.queued.replace("{n}", pos.toLocaleString("en-US")).replace("{wait}", wait);
  }

  function copyButton(id: string): HTMLButtonElement {
    const btn = h("button", { class: "btn btn-ghost btn-small", type: "button" }, copy.mine.copy) as HTMLButtonElement;
    btn.addEventListener("click", () => {
      void navigator.clipboard?.writeText(shareUrl(id)).then(() => {
        btn.textContent = copy.mine.copied;
        window.setTimeout(() => (btn.textContent = copy.mine.copy), 1500);
      });
    });
    return btn;
  }

  function render() {
    clear(list);
    el.hidden = items.length === 0;
    for (const q of items) {
      const link = shareUrl(q.id);
      const actions: HTMLElement[] = [];
      if (q.status === "answered") {
        const watch = h("button", { class: "btn btn-ghost btn-small", type: "button" }, copy.mine.watch);
        watch.addEventListener("click", () => opts.onWatch(q.id));
        actions.push(watch);
      }
      if (q.status !== "rejected") actions.push(copyButton(q.id));
      list.appendChild(
        h("li", { class: `my-question my-question-${q.status}` }, [
          h("p", { class: "my-question-text" }, `“${q.question}”`),
          h("p", { class: "my-question-status" }, statusLine(q)),
          ...(q.joined && q.status === "queued" ? [h("p", { class: "helper-text" }, copy.mine.joined)] : []),
          ...(q.status === "queued" || q.status === "thinking" ? [h("p", { class: "helper-text" }, copy.mine.keep_link)] : []),
          h("div", { class: "my-question-actions" }, [h("a", { class: "my-question-link label-mono", href: `/?a=${encodeURIComponent(q.id)}` }, link.replace(/^https?:\/\//, "")), ...actions]),
        ]),
      );
    }
  }

  function update(id: string, patch: Partial<MyQuestion>) {
    const current = items.find((q) => q.id === id);
    if (!current) return;
    items = saveMyQuestion({ ...current, ...patch });
    render();
  }

  // Questions still waiting: ask the server where each one stands. On open, then every few
  // minutes (a question can be moderated after it joined the line, and the live position is an
  // estimate between these checks). One small uncached request per waiting question per tab.
  function refreshWaiting() {
    for (const q of items.filter((x) => x.status === "queued" || x.status === "thinking")) refreshOne(q);
  }
  function refreshOne(q: MyQuestion) {
    getAnswerLookup(q.id)
      .then((row) => {
        if (disposed) return;
        if (row.status === "answered" && row.answer) {
          update(q.id, { status: "answered", answer: row.answer.answer });
          opts.onFoundAnswered(q.id);
        } else if (row.status === "thinking") {
          update(q.id, { status: "thinking" });
        } else if (row.status === "queued" && row.position !== undefined && row.claimed_total !== undefined) {
          update(q.id, { status: "queued", position: row.position, claimedAt: row.claimed_total });
        } else if (row.status === "rejected") {
          update(q.id, { status: "rejected", rejectedMessage: row.message });
        }
      })
      .catch(() => undefined); // a purged or unknown id just keeps its last known state
  }
  refreshWaiting();
  const refreshTimer = window.setInterval(refreshWaiting, REFRESH_MS);
  render();

  return {
    el,
    isMine: (id) => items.some((q) => q.id === String(id)),
    add(q) {
      items = saveMyQuestion(q);
      render();
    },
    update,
    onQueue(next) {
      tick = next;
      if (items.some((q) => q.status === "queued")) render();
    },
    refresh: refreshWaiting,
    dispose() {
      disposed = true;
      window.clearInterval(refreshTimer);
    },
  };
}
