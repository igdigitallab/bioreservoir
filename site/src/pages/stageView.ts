// One place on the page where the fly is seen working (2026-09-19: "when the page loads
// the fly must already be answering someone's question"). A persistent view with ONE 3D fly that
// shows, in turn, a question being simulated ("deciding"), an answer told in order (the question,
// the fly turning toward YES or NO while the brain replays the real spikes, then the verdict), or
// an election question whose verdict stays sealed. liveStage.ts decides WHAT to show and when;
// this module only knows HOW. The share page uses it for a single answer.
import { h, clear } from "../dom";
import { copy } from "../content";
import { createFlyIcon, type FlyIcon } from "../flyIcon";
import { stateToAnimations } from "../states";
import { TOTAL_REPLAY_MS } from "../frames";
import { buildAnswerCard } from "./answerCard";
import { buildLabReadout } from "./labReadout";
import type { Answer } from "../api/types";
import type { FlySide } from "../states";

/** Beat start times in ms after an answer is shown: pure, so the order is testable. */
export function answerSchedule(reducedMotion: boolean): { fly: number; verdict: number } {
  if (reducedMotion) return { fly: 0, verdict: 0 };
  const fly = 700; // long enough to read a short question before anything moves
  return { fly, verdict: fly + TOTAL_REPLAY_MS + 300 };
}

/** "12 s", "1 min 5 s": elapsed simulation time on the deciding caption. */
export function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s} s`;
  return `${Math.floor(s / 60)} min ${s % 60} s`;
}

export interface ShowAnswerOptions {
  label: string;
  /** Beat 2: the fly starts turning; the caller replays the brain here. */
  onFlyBeat?: () => void;
  /** Beat 3: the verdict is on screen. */
  onVerdict?: () => void;
}

export interface StageView {
  el: HTMLElement;
  /** The "Lab readout" panel, kept OUT of `el` on purpose (2026-09-21): it is a wide thing —
   * a five-column trial table, a 640px spike raster, a cell-type table — and `el` is the stage's
   * narrow left column (~315px on a 1440px screen, beside the brain). Rendered in there it
   * overflowed its own card: the "trial 1" column was clipped by the card edge, the bias rows
   * ran past the border, and the raster became a horizontal scroll showing three dots. The page
   * places this node across the full width of the stage panel instead. */
  readoutEl: HTMLElement;
  /** A question the worker is simulating right now; the caption counts from `startedAt`.
   * `yesSide` (known from the wording alone, api.py `pipeline.yes_side_for`) puts both answer
   * labels on their real sides while it is still being decided. */
  showDeciding: (question: string, startedAt: string | undefined, label: string, yesSide?: FlySide) => void;
  showAnswer: (answer: Answer, opts: ShowAnswerOptions) => void;
  /** An election question before Nov 4: the question is shown, the verdict is not. */
  showSealed: (question: string, label: string) => void;
  /** Nothing to show (empty queue, nothing answered yet). */
  showIdle: (text: string) => void;
  /** A question still in line (share page): the question, and where it stands. */
  showWaiting: (question: string, label: string, caption: string, yesSide?: FlySide) => void;
  /** The id currently on stage, if any. */
  currentId: () => string | undefined;
  dispose: () => void;
}

export function createStageView(opts: { compact?: boolean } = {}): StageView {
  const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const label = h("p", { class: "label-mono stage-label" }, "");
  const question = h("p", { class: "stage-question" }, "");
  const flyHost = h("div", { class: "fly-panel stage-fly", "aria-label": "The fly, turning toward its answer" });
  const caption = h("p", { class: "stage-caption", "aria-live": "polite" }, "");
  const verdict = h("div", { class: "stage-verdict", hidden: "" });
  const readoutEl = h("div", { class: "stage-readout", hidden: "" });
  const el = h("div", { class: `stage-view${opts.compact ? " stage-view-compact" : ""}` }, [label, question, flyHost, caption, verdict]);

  const fly: FlyIcon = createFlyIcon(flyHost, { holdAnswer: true });
  let timers: number[] = [];
  let ticker: number | undefined;
  let current: string | undefined;

  function reset(id: string | undefined) {
    timers.forEach((t) => window.clearTimeout(t));
    timers = [];
    if (ticker !== undefined) window.clearInterval(ticker);
    ticker = undefined;
    current = id;
    verdict.hidden = true;
    clear(verdict);
    readoutEl.hidden = true;
    clear(readoutEl);
    el.classList.remove("is-sealed");
  }

  /** Returns true when this is an update to the item already on screen (a queue tick): callers
   * use it to leave the fly alone, exactly as the text is left un-faded. */
  function setText(labelText: string, questionText: string, captionText: string): boolean {
    const quoted = questionText ? `“${questionText}”` : "";
    const same = question.textContent === quoted && label.textContent === labelText;
    label.textContent = labelText;
    question.textContent = quoted;
    question.hidden = !questionText;
    caption.textContent = captionText;
    if (same) return true;
    for (const node of [question, caption]) {
      node.classList.remove("reveal-in");
      void node.offsetWidth; // restart the fade for the new item
      node.classList.add("reveal-in");
    }
    return false;
  }

  /** Start (or keep) the "this question is being simulated" state on the fly. Restarting it on
   * every queue tick would blink the labels off and reset the wander, so a repeat is a no-op. */
  function startDeliberation(same: boolean, yesSide: FlySide | undefined) {
    if (same) return;
    fly.release();
    if (yesSide) fly.deliberate(yesSide);
  }

  function showDeciding(q: string, startedAt: string | undefined, labelText: string, yesSide?: FlySide) {
    reset(undefined);
    const since = startedAt ? Date.parse(startedAt) : Date.now();
    const tick = () => {
      caption.textContent = `${copy.stage.deciding} ${formatElapsed((Date.now() - since) / 1000)}`;
    };
    startDeliberation(setText(labelText, q, ""), yesSide);
    tick();
    ticker = window.setInterval(tick, 1000);
  }

  function showAnswer(answer: Answer, o: ShowAnswerOptions) {
    reset(answer.id);
    fly.release();
    // The question and both options are on screen from the first frame; the fly is already
    // looking around, and only starts its real turn on the fly beat (2026-09-19 design note).
    fly.deliberate(answer.lab.yes_side);
    const when = answerSchedule(reduced());
    const word = answer.answer === "yes" ? copy.answer.yes_label : copy.answer.no_label;
    setText(o.label, answer.question, copy.stage.deciding_short);
    timers.push(
      window.setTimeout(() => {
        fly.play(stateToAnimations(answer), TOTAL_REPLAY_MS);
        o.onFlyBeat?.();
      }, when.fly),
      window.setTimeout(() => {
        caption.textContent = copy.stage.turned.replace("{answer}", word);
        verdict.appendChild(buildAnswerCard(answer, { showQuestion: false }));
        readoutEl.appendChild(buildLabReadout(answer));
        verdict.hidden = false;
        readoutEl.hidden = false;
        verdict.classList.add("reveal-in");
        o.onVerdict?.();
      }, when.verdict),
    );
  }

  function showSealed(q: string, labelText: string) {
    reset(undefined);
    fly.release();
    setText(labelText, q, copy.stage.sealed);
    el.classList.add("is-sealed");
  }

  function showIdle(text: string) {
    reset(undefined);
    fly.release();
    setText(copy.stage.label_live, "", text);
  }

  function showWaiting(q: string, labelText: string, captionText: string, yesSide?: FlySide) {
    reset(undefined);
    // The share page calls this on every queue tick with a new position in the caption.
    startDeliberation(setText(labelText, q, captionText), yesSide);
  }

  return {
    el,
    readoutEl,
    showWaiting,
    showDeciding,
    showAnswer,
    showSealed,
    showIdle,
    currentId: () => current,
    dispose() {
      reset(undefined);
      fly.dispose();
    },
  };
}
