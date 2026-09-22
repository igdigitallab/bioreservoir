// The archive of everything the fly has been asked (2026-09-19: "a big list of previous
// questions, people give them likes, and the most liked ones stay around in their own tab").
// Two tabs over the same list: Latest (newest first) and Most liked. Answers the stage has just
// shown are prepended to Latest live; older ones arrive a page at a time from GET /api/questions.
//
// A like is one per client per question — the server's likes table enforces that, this module
// only remembers which ones THIS browser liked so the hearts stay filled after a reload.
import { h, clear } from "../dom";
import { copy } from "../content";
import { getQuestions, likeAnswer } from "../api/client";
import { fmtTime } from "../format";
import { localStore, type KeyValueStore } from "../storage";
import type { AnswerSummary } from "../api/types";

export const PAGE_SIZE = 20;
/** A tab left open for hours must not grow a DOM row per answer forever. */
export const MAX_ROWS = 200;
const STORAGE_NAME = "bioreservoir.liked.v1";

export function loadLiked(store: KeyValueStore | undefined = localStore()): Set<string> {
  if (!store) return new Set();
  try {
    const parsed: unknown = JSON.parse(store.getItem(STORAGE_NAME) ?? "[]");
    return new Set(Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === "string") : []);
  } catch {
    return new Set();
  }
}

export function saveLiked(ids: Set<string>, store: KeyValueStore | undefined = localStore()): void {
  try {
    // Only the most recent few hundred: this is a UI hint, not a record worth growing forever.
    store?.setItem(STORAGE_NAME, JSON.stringify([...ids].slice(-500)));
  } catch {
    // quota or blocked storage: the heart just won't survive a reload
  }
}

/** Merge a live answer into a list already on screen: newest first, never twice. Pure, so the
 * "the stage showed it, put it on top" path is testable without a DOM. */
export function mergeNewest(list: AnswerSummary[], incoming: AnswerSummary): AnswerSummary[] {
  const id = String(incoming.id);
  return [incoming, ...list.filter((a) => String(a.id) !== id)];
}

/** What the row's verdict column says. An embargoed answer is held back for everyone but its
 * asker, exactly as on the stage — the question is public, the answer is not. */
export function verdictLabel(a: AnswerSummary): string {
  if (a.embargoed || a.answer === null) return copy.recent.sealed;
  return a.answer === "yes" ? copy.answer.yes_label : copy.answer.no_label;
}

export interface QuestionsPanel {
  el: HTMLElement;
  /** An answer the stage has just finished showing: it belongs at the top of Latest. */
  prepend: (summary: AnswerSummary) => void;
}

type Tab = "recent" | "top";

export function mountQuestions(): QuestionsPanel {
  const liked = loadLiked();
  const lists: Record<Tab, AnswerSummary[]> = { recent: [], top: [] };
  const loaded: Record<Tab, boolean> = { recent: false, top: false };
  const hasMore: Record<Tab, boolean> = { recent: false, top: false };
  /** How far into the SERVER's ordering this tab has read. Counted separately from the rows on
   * screen: "most liked" reorders while you read it, so a page can come back entirely of rows we
   * already have — paging by list length would then ask for the same offset forever. */
  const fetched: Record<Tab, number> = { recent: 0, top: 0 };
  let tab: Tab = "recent";
  let busy = false;
  /** How many answers exist in total — from the server, then +1 for each live one, so the count
   * on screen never lags behind the rows under it. */
  let totalCount = 0;

  const listEl = h("ul", { class: "recent-list questions-list" });
  const status = h("p", { class: "helper-text" }, copy.questions.loading);
  const more = h("button", { class: "btn btn-ghost questions-more", type: "button" }, copy.questions.more) as HTMLButtonElement;
  const total = h("span", { class: "label-mono questions-total" }, "");
  const tabs = (["recent", "top"] as const).map((name) =>
    h("button", { class: "questions-tab", type: "button", "data-tab": name },
      name === "recent" ? copy.questions.tab_recent : copy.questions.tab_top) as HTMLButtonElement,
  );
  const el = h("div", { class: "panel recent-panel questions-panel" }, [
    h("div", { class: "questions-head" }, [
      h("h3", { class: "label-mono" }, copy.questions.title),
      h("div", { class: "questions-tabs", role: "tablist" }, tabs),
    ]),
    total,
    status,
    listEl,
    more,
  ]);

  function likeButton(a: AnswerSummary): HTMLButtonElement {
    const id = String(a.id);
    const count = h("span", { class: "like-count" }, String(a.likes ?? 0));
    const btn = h("button", {
      class: `like-btn${liked.has(id) ? " is-liked" : ""}`,
      type: "button",
      "aria-label": copy.questions.like,
      "aria-pressed": String(liked.has(id)),
    }, [h("span", { class: "like-heart", "aria-hidden": "true" }, "♥"), count]) as HTMLButtonElement;
    btn.addEventListener("click", (event) => {
      event.preventDefault();
      if (btn.disabled) return;
      btn.disabled = true;
      likeAnswer(id)
        .then((res) => {
          // The server owns both numbers: a client whose storage was cleared toggles its real
          // like off rather than adding a second one, and the count it reports is the truth.
          if (res.liked) liked.add(id);
          else liked.delete(id);
          saveLiked(liked);
          for (const list of [lists.recent, lists.top]) {
            const found = list.find((item) => String(item.id) === id);
            if (found) found.likes = res.likes;
          }
          count.textContent = String(res.likes);
          btn.classList.toggle("is-liked", res.liked);
          btn.setAttribute("aria-pressed", String(res.liked));
        })
        .catch((err) => {
          // Rate-limited or offline: say so on the button instead of leaving a tap that did
          // nothing visible.
          console.error("like failed", err);
          btn.title = copy.questions.like_failed;
          btn.classList.add("is-failed");
          window.setTimeout(() => {
            btn.classList.remove("is-failed");
            btn.title = "";
          }, 2500);
        })
        .finally(() => {
          btn.disabled = false;
        });
    });
    return btn;
  }

  function renderTotal() {
    total.textContent = copy.questions.total.replace("{n}", totalCount.toLocaleString("en-US"));
  }

  function render() {
    clear(listEl);
    const list = lists[tab];
    status.textContent = loaded[tab] && list.length === 0 ? copy.questions.empty : "";
    status.hidden = status.textContent === "";
    for (const a of list) {
      listEl.appendChild(
        h("li", {}, [
          h("a", { href: `/?a=${encodeURIComponent(String(a.id))}` },
            h("span", { class: "feed-question" }, `“${a.question}”`)),
          // The time, the verdict and the heart are one tail group, a sibling of the link rather
          // than half in and half out of it (2026-09-21): on a phone the question wraps to two or
          // three lines, and while the meta lived inside the link it stayed pinned to the FIRST
          // line while the heart centred itself over all of them — every row ragged in a
          // different place. As one group they wrap together onto a line of their own.
          h("div", { class: "feed-meta" }, [
            h("span", { class: "label-mono" }, fmtTime(a.answered_at)),
            h("span", { class: "label-mono feed-verdict" }, verdictLabel(a)),
            likeButton(a),
          ]),
        ]),
      );
    }
    more.hidden = !hasMore[tab];
    for (const button of tabs) button.classList.toggle("is-active", button.dataset.tab === tab);
  }

  function load(next: boolean) {
    if (busy) return;
    busy = true;
    const target = tab;
    const offset = next ? fetched[target] : 0;
    more.disabled = true;
    getQuestions(target, PAGE_SIZE, offset)
      .then((page) => {
        // Merge, never replace: a live answer prepended while this request was in flight stays.
        const merged = next ? [...lists[target], ...page.items] : page.items.slice();
        const byId = new Map(merged.map((a) => [String(a.id), a]));
        for (const a of lists[target]) if (!byId.has(String(a.id))) byId.set(String(a.id), a);
        lists[target] = target === "top"
          ? [...byId.values()].sort((a, b) => (b.likes ?? 0) - (a.likes ?? 0) || Number(b.id) - Number(a.id))
          : [...byId.values()].sort((a, b) => Number(b.id) - Number(a.id));
        loaded[target] = true;
        fetched[target] = offset + page.items.length;
        // Nothing new despite the server saying there is more (a reordered "most liked" page of
        // rows we already hold): stop offering the button instead of looping on it.
        hasMore[target] = page.has_more && page.items.length > 0;
        lists[target] = lists[target].slice(0, MAX_ROWS);
        totalCount = Math.max(page.total, lists[target].length);
        renderTotal();
        if (target === tab) render();
      })
      .catch((err) => {
        console.error("questions fetch failed", err);
        if (!loaded[target]) {
          status.textContent = copy.questions.offline;
          status.hidden = false;
        }
      })
      .finally(() => {
        busy = false;
        more.disabled = false;
      });
  }

  for (const button of tabs) {
    button.addEventListener("click", () => {
      const next = button.dataset.tab === "top" ? "top" : "recent";
      if (next === tab) return;
      tab = next;
      render();
      // The most-liked order changes while people are on the page, so reload it on every visit
      // to the tab; Latest is kept live by `prepend`.
      if (!loaded[tab] || tab === "top") {
        // A reload of "most liked" starts the server ordering over, so the rows it already holds
        // must go too — otherwise the next "Show more" jumps past everything in between.
        if (tab === "top") {
          lists.top = [];
          fetched.top = 0;
        }
        load(false);
      }
    });
  }
  more.addEventListener("click", () => load(true));

  load(false);

  return {
    el,
    prepend(summary) {
      const before = lists.recent.length;
      lists.recent = mergeNewest(lists.recent, summary);
      loaded.recent = true;
      if (lists.recent.length > before) {
        totalCount += 1;
        fetched.recent += 1; // this row came from the front of the server's ordering too
        renderTotal();
      }
      lists.recent = lists.recent.slice(0, MAX_ROWS);
      if (tab === "recent") render();
    },
  };
}
