// The page every shared link lands on (`/a/{id}` → `/?a={id}`). Question first, then the fly
// turning, then the verdict: the same told-in-order sequence as the live stage (2026-09-19, all
// three design reviews: the old brain-first card buried the answer the visitor clicked for). A
// question still in line shows its live place and an honest wait, and plays here the moment the
// fly answers it, so "keep this link" works for a wait of minutes or days.
import { h, clear } from "../dom";
import { copy } from "../content";
import { getAnswer, getAnswerLookup, subscribeEvents } from "../api/client";
import { createBrainScene, type BrainScene } from "../scene";
import { REPLAY_STRETCH } from "../frames";
import { formatWait, livePosition, loadMyQuestions } from "../queue";
import { createStageView } from "./stageView";
import { buildAboutLab } from "./aboutLab";
import { buildNav } from "./nav";
import { buildFooter } from "./footer";
import type { Answer, AnswerLookup, ServerEvent } from "../api/types";

export function mountShare(root: HTMLElement, id: string): () => void {
  clear(root);
  root.appendChild(buildNav());
  const view = createStageView();
  const heroCanvas = h("div", { class: "hero-canvas", role: "img", "aria-label": "Replay of this answer's real spikes" });
  const brain = h("div", { class: "stage-brain" }, [heroCanvas, h("span", { class: "replay-tag stage-tag" }, `real spikes, slowed ${REPLAY_STRETCH}×`)]);
  const cta = h("div", { class: "share-cta" }, [
    h("a", { class: "btn", href: "/#ask-box" }, "Ask the fly your own question"),
    h("a", { class: "btn btn-ghost", href: "/#live" }, "Watch it answer live"),
  ]);
  root.appendChild(
    h("section", { class: "container share-page" }, [h("div", { class: "panel stage-panel" }, [view.el, brain, view.readoutEl]), cta]),
  );
  root.appendChild(h("div", { class: "container" }, buildAboutLab()));
  root.appendChild(buildFooter());
  const scene: BrainScene = createBrainScene(heroCanvas, { interactive: true, showVnc: true });

  const mine = loadMyQuestions().some((q) => q.id === String(id));
  let row: AnswerLookup | undefined;
  let played = false;
  let disposed = false;

  function play(answer: Answer, embargoed: boolean) {
    if (played || disposed) return;
    played = true;
    if (embargoed && !mine) {
      view.showSealed(answer.question, copy.stage.label_shared);
      return;
    }
    view.showAnswer(answer, {
      label: mine ? copy.stage.label_yours : copy.stage.label_shared,
      onFlyBeat: () => scene.playAnswer(answer),
    });
  }

  function showWaiting(claimedNow?: number, avgCycleS?: number) {
    if (!row || row.status !== "queued" || row.position === undefined) return;
    const pos = claimedNow !== undefined && row.claimed_total !== undefined ? livePosition(row.position, row.claimed_total, claimedNow) : row.position;
    const wait = formatWait(pos, avgCycleS ?? row.avg_cycle_s ?? NaN);
    view.showWaiting(
      row.question,
      `In line · #${pos.toLocaleString("en-US")}`,
      `${wait} at the current pace. ${copy.mine.keep_link}`,
      row.yes_side,
    );
  }

  function load() {
    getAnswerLookup(id)
      .then((next) => {
        if (disposed) return;
        row = next;
        if (next.status === "answered" && next.answer) play(next.answer, !!next.embargoed);
        // Answered, embargoed, and the server did not send the verdict: this visitor is not the
        // asker (api.py `get_answer`). The question is public, its answer is not — yet.
        else if (next.status === "answered" && next.embargoed) {
          played = true;
          view.showSealed(next.question, copy.stage.label_shared);
        }
        else if (next.status === "thinking") view.showDeciding(next.question, next.started_at, copy.stage.label_deciding, next.yes_side);
        else if (next.status === "queued") showWaiting();
        else view.showIdle("This question was not answered.");
      })
      .catch(() => view.showIdle("This answer could not be found."));
  }

  function handle(event: ServerEvent) {
    if (played) return;
    if (event.type === "queue") showWaiting(event.data.claimed_total, event.data.avg_cycle_s);
    else if (event.type === "thinking" && String(event.data.id) === String(id)) view.showDeciding(event.data.question, event.data.started_at, copy.stage.label_deciding, event.data.yes_side);
    else if (event.type === "answered" && String(event.data.id) === String(id)) {
      getAnswer(id)
        .then((answer) => play(answer, event.data.embargoed))
        .catch(() => {
          // A non-asker gets no verdict for an embargoed question (api.py `get_answer`): show the
          // sealed card rather than reloading into the same wall.
          if (event.data.embargoed && !mine) {
            played = true;
            view.showSealed(event.data.question, copy.stage.label_shared);
          } else load();
        });
    } else if (event.type === "reconnected") load();
  }

  view.showIdle(copy.stage.loading);
  load();
  const unsubscribe = subscribeEvents(handle);

  return () => {
    disposed = true;
    unsubscribe();
    view.dispose();
    scene.dispose();
  };
}
