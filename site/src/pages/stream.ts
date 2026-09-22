// The /stream layout: fixed 1920x1080 for a headless-Chrome capture -> ffmpeg -> RTMP pipeline
// (see docs/ROADMAP.md stage B). No interactive controls, no click handlers — this page is only
// ever looked at, never touched. Three columns: brain + fly icon, the real spike raster +
// decision math (the central "prove it's real" element, falls back to a static explainer when
// no answer has been seen yet), live stats + the public chat slot; a fixed-height footer row for
// the ticker + QR so the whole thing fits exactly in 1920x1080, no scrollbars.
//
// Kept as the dark broadcast variant on purpose (2026-09-18 decision, reaffirmed 2026-09-19): a
// YouTube/Twitch stream sitting on a light page is not this component's job to fix, just its
// contrast, empty states and duplicate messaging.
import QRCode from "qrcode";
import { h, clear } from "../dom";
import { copy } from "../content";
import { createBrainScene, type BrainScene } from "../scene";
import { createFlyIcon, type FlyIcon } from "../flyIcon";
import { getAnswer, getFeed, getStats, subscribeEvents } from "../api/client";
import { mountChat } from "./chat"; // CHAT-INTEGRATION: see mount point below (read-only variant)
import { buildDecisionMath } from "../raster";
import { buildRasterView } from "./rasterView";
import { fmtNum, fmtSigned, fmtStr, DASH } from "../format";
import { TOTAL_REPLAY_MS } from "../frames";
import { stateToAnimations } from "../states";
import type { Answer, AnswerSummary, ServerEvent, Stats } from "../api/types";

const STATS_POLL_MS = 30000;

export function mountStream(root: HTMLElement): () => void {
  clear(root);
  document.body.classList.add("stream-mode");

  const brainSlot = h("div", { class: "hero-canvas stream-brain" });
  const flyPanelHost = h("div", { class: "fly-panel stream-fly-panel", "aria-label": "Top-down fly, current behavioural state" });
  const flyCard = h("div", { class: "stream-fly-card" }, flyPanelHost);
  const question = h("div", { class: "stream-question" }, "Waiting for the next question…");
  const rasterSlot = h("div", { class: "stream-raster-slot" });
  const decisionSlot = h("div", { class: "stream-decision" });
  const lastAnswerLine = h("p", { class: "stream-last-answer" }, "");
  const statsSlot = h("div", { class: "stream-stats" });
  // Task brief "Live chat slot": mounted by a separate agent, not built here. The right column
  // is entirely this slot's job now — it used to be a "RECENT ANSWERS" list floating over ~800px
  // of empty dark canvas (2026-09-19 audit); that list moved to `lastAnswerLine` above, a single
  // line of context next to the current answer instead of its own mostly-empty column.
  const chatSlot = h("aside", { class: "live-chat-slot stream-chat-slot", id: "live-chat-slot" });
  // The track holds the ticker text twice back-to-back and the CSS animation slides it exactly
  // -50% — standard seamless-marquee technique (style.css `.stream-ticker-track`). A ticker that
  // never moves just truncates mid-sentence permanently (2026-09-19 audit: "clipped mid-word").
  const tickerText = copy.stream.ticker.join("   •   ");
  const tickerTrack = h("div", { class: "stream-ticker-track" }, `${tickerText}     •     ${tickerText}`);
  const qrImg = h("img", { alt: "QR code to ask a question", width: "80", height: "80" });
  const publicUrl = import.meta.env.VITE_PUBLIC_URL || window.location.origin;
  const qrBlock = h("div", { class: "stream-qr" }, [
    qrImg,
    h("div", {}, [h("p", {}, copy.stream.qr_caption), h("p", { class: "label-mono" }, publicUrl)]),
  ]);

  // CHAT-INTEGRATION: read-only livestream chat (docs/LIVE.md "Live chat"), dark variant of the same
  // component home.ts mounts interactively -- this is the ONE place it is mounted on this page.
  const chat = mountChat({ readOnly: true });

  const page = h("div", { class: "stream-page" }, [
    h("div", { class: "stream-col stream-col-brain" }, [h("h2", {}, copy.stream.headline), brainSlot, flyCard]),
    h("div", { class: "stream-col stream-col-center" }, [question, rasterSlot, decisionSlot, lastAnswerLine]),
    h("div", { class: "stream-col stream-col-side" }, [statsSlot, chatSlot]),
    h("div", { class: "stream-footer" }, [h("div", { class: "stream-ticker" }, tickerTrack), qrBlock]),
  ]);
  root.appendChild(page);
  // Read-only chat in the right-hand column, under the live counters.
  chatSlot.appendChild(chat.el);

  QRCode.toDataURL(publicUrl, { margin: 1, color: { dark: "#0c0a09", light: "#ffffff" } })
    .then((url) => qrImg.setAttribute("src", url))
    .catch((err) => console.error("qr generation failed", err));

  const scene: BrainScene = createBrainScene(brainSlot, { interactive: false, showVnc: false });
  const flyIcon: FlyIcon = createFlyIcon(flyPanelHost);

  /** Single line of context for the answer BEFORE this one — not a "recent answers" list (that
   * real estate now goes to the chat slot), so only the immediately-previous answer, never the
   * current one. */
  function renderLastAnswer(answers: AnswerSummary[], currentId: string | undefined) {
    const prev = answers.find((a) => a.id !== currentId);
    const verdict = (a: AnswerSummary) => (a.answer === null ? copy.recent.sealed : a.answer === "yes" ? copy.answer.yes_label : copy.answer.no_label);
    lastAnswerLine.textContent = prev ? `Previously: “${prev.question}” (${verdict(prev)})` : "";
  }

  /** An election question before Nov 4: its verdict stays off the broadcast (embargo). */
  function renderSealed() {
    rasterSlot.classList.add("is-placeholder");
    clear(rasterSlot);
    clear(decisionSlot);
    decisionSlot.appendChild(h("p", { class: "label-mono" }, copy.stage.sealed));
  }

  /** SSE and the feed carry summaries; the full answer (spike replay) comes from the cached route. */
  function play(summary: AnswerSummary, replay: boolean) {
    question.textContent = summary.question;
    if (summary.embargoed) {
      renderSealed();
      return;
    }
    getAnswer(summary.id)
      .then((answer: Answer) => {
        if (replay) {
          scene.playAnswer(answer);
          flyIcon.play(stateToAnimations(answer), TOTAL_REPLAY_MS);
        }
        renderCurrentAnswer(answer);
      })
      .catch((err) => console.error("answer fetch failed", err));
  }

  /** Renders synchronously with dash placeholders so this card is never a content-less blank
   * white rectangle before the first `getStats()` resolves (2026-09-19 audit: "unexplained empty
   * white bar top-right" — that bar was this card, empty, with only its own padding visible). */
  function renderStats(stats: Stats | null) {
    clear(statsSlot);
    statsSlot.appendChild(h("h3", { class: "label-mono" }, "Live counters"));
    statsSlot.appendChild(
      h("div", { class: "stream-stat-grid" }, [
        h("div", {}, [h("span", { class: "stream-stat-value" }, stats ? fmtNum(stats.answered) : DASH), h("span", {}, " answered")]),
        h("div", {}, [h("span", { class: "stream-stat-value" }, stats ? `${fmtNum(stats.yes)}/${fmtNum(stats.no)}` : DASH), h("span", {}, " yes/no")]),
        h("div", {}, [h("span", { class: "stream-stat-value" }, stats ? fmtNum(stats.total_spikes) : DASH), h("span", {}, " real spikes")]),
      ]),
    );
  }

  function refreshStats() {
    getStats()
      .then(renderStats)
      .catch((err) => {
        console.error("getStats failed", err);
      });
  }

  /** No answer has been seen yet this session — a static walk-through instead of an empty box.
   * Kept to ONE state message (`question`'s own "Waiting for the next question…" above already
   * says this — 2026-09-19 audit flagged the old second "WAITING FOR THE FIRST QUESTION..." line
   * here as a duplicate), so this one instead explains what will appear once a question lands. */
  function renderExplainer() {
    rasterSlot.classList.add("is-placeholder");
    clear(rasterSlot);
    rasterSlot.appendChild(
      h(
        "ol",
        { class: "stream-explainer" },
        copy.how_it_works.map((step, i) => h("li", {}, [h("span", { class: "stream-explainer-step" }, `${i + 1}`), h("span", {}, `${step.title} — ${step.body}`)])),
      ),
    );
    clear(decisionSlot);
    decisionSlot.appendChild(h("p", { class: "label-mono" }, "Real spikes and the decision math appear here once a question is answered."));
  }

  function renderCurrentAnswer(answer: Answer) {
    rasterSlot.classList.remove("is-placeholder");
    clear(rasterSlot);
    const raster = buildRasterView(answer.lab, { width: 760, interactive: false });
    rasterSlot.appendChild(raster.el);

    clear(decisionSlot);
    const math = buildDecisionMath(answer.lab);
    decisionSlot.appendChild(
      h("div", { class: "decision-big" }, [
        h("div", { class: "decision-big-row" }, [h("span", {}, "raw bias"), h("span", {}, fmtSigned(math.meanRawBias))]),
        h("div", { class: "decision-big-row" }, [h("span", {}, "− handedness"), h("span", {}, fmtSigned(math.b0))]),
        h("div", { class: "decision-big-row decision-big-result" }, [h("span", {}, "= corrected bias"), h("span", {}, fmtSigned(math.correctedBias))]),
        h("div", { class: "decision-big-row decision-big-result" }, [h("span", {}, "→ turned"), h("span", {}, fmtStr(math.yesSide))]),
      ]),
    );
  }

  renderStats(null);
  renderExplainer();
  getFeed()
    .then((feed) => {
      // Seed the center panel with the last real answer instead of leaving the explainer up
      // forever once the session actually has data.
      const last = feed.recent[0];
      if (last) play(last, false);
      renderLastAnswer(feed.recent, last?.id);
    })
    .catch((err) => console.error("feed fetch failed", err));
  refreshStats();
  const statsTimer = window.setInterval(refreshStats, STATS_POLL_MS);

  // A passive broadcast screen (OBS, a physical display): every answer plays as it lands.
  function handleServerEvent(event: ServerEvent) {
    if (event.type === "thinking") {
      question.textContent = event.data.question;
      // Same as the live stage: both options up and the fly looking around while it is decided.
      if (event.data.yes_side) flyIcon.deliberate(event.data.yes_side);
    } else if (event.type === "answered") {
      play(event.data, true);
      refreshStats();
      getFeed()
        .then((feed) => renderLastAnswer(feed.recent, event.data.id))
        .catch(() => undefined);
    }
  }
  const unsubscribe = subscribeEvents(handleServerEvent);

  return () => {
    document.body.classList.remove("stream-mode");
    clearInterval(statsTimer);
    unsubscribe();
    scene.dispose();
    flyIcon.dispose();
    chat.dispose(); // CHAT-INTEGRATION
  };
}
