import { h, clear } from "../dom";
import { copy } from "../content";
import { renderHeadline } from "../headline";
import { createBrainScene, type BrainScene } from "../scene";
import { mountTurnstile, TURNSTILE_STATUS_TEXT, type TurnstileHandle, type TurnstileStatus } from "../turnstile";
import { askQuestion, getAnswer, subscribeEvents } from "../api/client";
import { mountLiveStage } from "./liveStage";
import { mountMyQuestions } from "./myQuestions";
import { relativeAgo } from "./liveStage";
import { mountLabStats } from "./labStats";
import { mountValidation } from "./validation";
import { buildAboutLab } from "./aboutLab";
import { buildNav } from "./nav";
import { buildFooter } from "./footer";
import { mountChat, type ChatHandle } from "./chat";
import { mountQuestions } from "./questions";
import { REPLAY_STRETCH } from "../frames";
import type { AnswerSummary, ServerEvent } from "../api/types";

const MAX_LEN = 140;
/** A feed row normally appears when its answer's verdict is shown on the stage; if the stage never
 * gets to it (a burst), it appears anyway after this long. */
const FEED_FALLBACK_MS = 90_000;

// Live page v2 (2026-09-19, design brief + three external reviews): built
// for a crowd that mostly waits. Order on a phone: headline and a two-line deck, the ask box, the
// visitor's own questions, the LIVE stage where the fly answers everyone's questions in turn, the
// recent answers, the chat, then the compact numbers and the explanations.
export function mountHome(root: HTMLElement): () => void {
  clear(root);

  // --- Ask box: compact, rules behind a toggle (they only matter to someone about to ask) ---
  const textarea = h("textarea", {
    maxlength: String(MAX_LEN),
    rows: "2",
    placeholder: copy.ask.placeholder,
    "aria-label": copy.ask.placeholder,
  }) as HTMLTextAreaElement;
  const charCount = h("span", { class: "char-count" }, `0/${MAX_LEN}`);
  const turnstileMount = h("div", { class: "turnstile-mount" });
  const submitBtn = h("button", { class: "btn", type: "button", disabled: "true" }, copy.hero.cta) as HTMLButtonElement;
  const turnstileStatus = h("div", { class: "turnstile-status label-mono", "aria-live": "polite" }, "");
  const rejection = h("div", { class: "rejection", "aria-live": "assertive" });
  const rules = h("details", { class: "ask-rules" }, [
    h("summary", {}, copy.ask.rules_toggle),
    h("p", { class: "helper-text" }, copy.ask.helper),
    h("ul", { class: "rules-list" }, copy.ask.rules.map((r) => h("li", {}, r))),
  ]);
  // What becomes public, and what an answer is not — said once, before anyone types, rather than
  // only in the footer and the Terms (2026-09-21).
  const askDisclaimer = h("p", { class: "ask-disclaimer helper-text" }, [
    copy.ask.disclaimer,
    " ",
    h("a", { href: "/terms" }, copy.ask.terms_note),
  ]);
  const askBox = h("div", { class: "panel ask-box", id: "ask-box" }, [
    textarea,
    turnstileMount,
    h("div", { class: "ask-row" }, [charCount, submitBtn]),
    turnstileStatus,
    rejection,
    askDisclaimer,
    rules,
  ]);

  // --- Brain (the real spikes replay here when the stage's fly turns) ---
  const heroCanvas = h("div", { class: "hero-canvas", role: "img", "aria-label": "3D render of the fly brain" });
  const brain = h("div", { class: "stage-brain" }, [
    heroCanvas,
    h("span", { class: "replay-tag stage-tag" }, `real spikes, slowed ${REPLAY_STRETCH}×`),
  ]);
  const scene: BrainScene = createBrainScene(heroCanvas, { interactive: true, showVnc: false });

  // --- The archive: every question ever answered, newest or most liked. A row for a brand-new
  // answer appears only once the stage has shown its verdict, never before. ---
  const questions = mountQuestions();
  const seen = new Set<string>();
  const waitingForStage = new Map<string, { summary: AnswerSummary; timer: number }>();

  function releaseFeedRow(id: string) {
    const entry = waitingForStage.get(String(id));
    if (!entry) return;
    window.clearTimeout(entry.timer);
    waitingForStage.delete(String(id));
    if (seen.has(String(id))) return;
    seen.add(String(id));
    questions.prepend(entry.summary);
    labStats.refresh();
  }

  // --- The visitor's own questions and the live stage ---
  const mine = mountMyQuestions({
    onWatch: (id) => playOwn(id, copy.stage.label_yours),
    onFoundAnswered: (id) => playOwn(id, copy.stage.label_yours),
  });
  const stage = mountLiveStage({
    onFlyBeat: (answer) => scene.playAnswer(answer),
    onShown: (id) => releaseFeedRow(id),
    isMine: (id) => mine.isMine(String(id)),
    onQueue: (tick) => mine.onQueue(tick),
  });
  const stagePanel = h("div", { class: "panel stage-panel", id: "live" }, [stage.statusEl, stage.el, brain, stage.readoutEl]);

  function playOwn(id: string, label: string) {
    getAnswer(id)
      .then((answer) => {
        mine.update(String(id), { status: "answered", answer: answer.answer });
        stage.playOwn(answer, label);
        stagePanel.scrollIntoView({ behavior: "smooth", block: "start" });
      })
      .catch((err) => console.error("own answer fetch failed", err));
  }

  // --- Layout ---
  const heroHead = h("header", { class: "hero-head" }, [
    h("h1", {}, renderHeadline(copy.hero.headline)),
    h("p", { class: "hero-deck" }, copy.hero.subheadline),
  ]);
  const chatSlot = h("aside", { class: "live-chat-slot", id: "live-chat-slot" });
  const liveLayout = h("section", { class: "hero container" }, [
    h("div", { class: "hero-row" }, [heroHead, h("div", { class: "ask-column" }, [askBox, mine.el])]),
    h("div", { class: "live-layout" }, [h("div", { class: "live-main" }, [stagePanel, questions.el]), chatSlot]),
  ]);

  const labStats = mountLabStats();
  const validation = mountValidation();
  const howGrid = h(
    "div",
    { class: "how-grid" },
    copy.how_it_works.map((item, i) =>
      h("div", { class: "panel how-card" }, [
        h("h3", {}, [h("span", { class: "how-card-index" }, String(i + 1)), item.title]),
        h("p", {}, item.body),
      ]),
    ),
  );
  const sections = h("div", { class: "container" }, [
    labStats.el,
    h("section", { class: "section", id: "how-it-works" }, [h("h2", {}, renderHeadline(copy.sections.how_it_works)), howGrid]),
    h("section", { class: "section honesty" }, [
      h("div", { class: "editorial-grid" }, [
        h("h2", {}, renderHeadline(copy.honesty.title)),
        h("div", { class: "editorial-grid-body" }, h("p", {}, copy.honesty.body)),
      ]),
    ]),
    validation.el,
    buildAboutLab(),
  ]);

  root.appendChild(buildNav());
  root.appendChild(liveLayout);
  root.appendChild(sections);
  root.appendChild(buildFooter());

  // The chat mounts last, into a slot the page already owns. It is the one panel here that
  // touches localStorage and a third-party widget, and until 2026-09-21 it was mounted BEFORE the
  // two appends above: one throw inside it (blocked storage — Safari private mode, an embedded
  // webview) silently deleted the entire lower half of the page, sections and footer included,
  // Terms and Privacy links with them. Order plus this catch make that impossible.
  let chat: ChatHandle | undefined;
  try {
    chat = mountChat();
    chatSlot.appendChild(chat.el);
  } catch (err) {
    console.error("chat failed to mount", err);
  }

  // --- Turnstile ---
  let turnstileHandle: TurnstileHandle | undefined;
  let currentToken: string | undefined;
  function setTurnstileStatus(s: TurnstileStatus) {
    turnstileStatus.textContent = TURNSTILE_STATUS_TEXT[s];
    if (s !== "verified") submitBtn.disabled = true;
  }
  mountTurnstile(
    turnstileMount,
    (token) => {
      currentToken = token;
      submitBtn.disabled = false;
    },
    setTurnstileStatus,
  )
    .then((handle) => {
      turnstileHandle = handle;
    })
    .catch((err) => {
      console.error("turnstile failed to load", err);
      turnstileStatus.textContent = "Verification widget failed to load — reload the page.";
    });

  textarea.addEventListener("input", () => {
    charCount.textContent = `${textarea.value.length}/${MAX_LEN}`;
  });

  // --- One SSE stream for the stage, the archive and the visitor's own questions ---
  function handleServerEvent(event: ServerEvent) {
    stage.handle(event);
    if (event.type === "reconnected") mine.refresh();
    if (event.type === "thinking" && mine.isMine(String(event.data.id))) {
      mine.update(String(event.data.id), { status: "thinking" });
    } else if (event.type === "answered") {
      const summary = event.data;
      const id = String(summary.id);
      if (!seen.has(id) && !waitingForStage.has(id)) {
        const timer = window.setTimeout(() => releaseFeedRow(id), FEED_FALLBACK_MS);
        waitingForStage.set(id, { summary, timer });
      }
      if (mine.isMine(id)) playOwn(id, copy.stage.label_yours);
    }
  }
  const unsubscribe = subscribeEvents(handleServerEvent);

  async function submit() {
    const question = textarea.value.trim();
    if (question.length === 0 || question.length > MAX_LEN) return;
    if (!currentToken) return;
    submitBtn.disabled = true;
    rejection.textContent = "";
    const response = await askQuestion(question, currentToken).catch((err) => {
      rejection.textContent = String(err);
      return undefined;
    });
    turnstileHandle?.reset();
    currentToken = undefined;
    if (!response) return;
    if (response.status === "rejected") {
      rejection.textContent = copy.ask.rejected_messages[response.reason] ?? response.message;
      return;
    }
    textarea.value = "";
    charCount.textContent = `0/${MAX_LEN}`;
    const id = String(response.id);
    const askedAt = new Date().toISOString();
    if (response.status === "answered") {
      // Asked before: the same words, the same seeds, the same answer. Shown at once, and labelled.
      mine.add({ id, question, status: "answered", position: 0, claimedAt: 0, askedAt, repeat: true });
      getAnswer(id)
        .then((answer) => {
          mine.update(id, { answer: answer.answer });
          stage.playOwn(answer, copy.stage.label_yours_repeat.replace("{when}", relativeAgo(answer.answered_at)));
          stagePanel.scrollIntoView({ behavior: "smooth", block: "start" });
        })
        .catch((err) => console.error("repeat answer fetch failed", err));
      return;
    }
    mine.add({
      id,
      question,
      status: "queued",
      position: response.position,
      claimedAt: response.claimed_total,
      askedAt,
      joined: response.joined,
    });
    mine.onQueue({ queue_length: response.position, claimed_total: response.claimed_total, avg_cycle_s: response.avg_cycle_s });
  }
  submitBtn.addEventListener("click", () => void submit());

  return () => {
    unsubscribe();
    stage.dispose();
    mine.dispose();
    scene.dispose();
    labStats.destroy();
    validation.destroy();
    chat?.dispose();
    for (const { timer } of waitingForStage.values()) window.clearTimeout(timer);
  };
}
