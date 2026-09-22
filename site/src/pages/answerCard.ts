import { h } from "../dom";
import { copy } from "../content";
import { shareUrl } from "../api/client";
import type { Answer } from "../api/types";

const STATE_KEYS: Array<keyof Answer["states"]> = [
  "appetite",
  "fear",
  "backoff",
  "courtship",
  "arousal",
];
// Mechanical UI labels, not editorial copy — content/copy.json's schema has no fields for them.
const SHARE_LABEL = "Share";

async function share(answer: Answer, button: HTMLButtonElement): Promise<void> {
  const url = shareUrl(answer.id);
  const text = copy.answer.share_text_template
    .replace("{question}", answer.question)
    .replace("{answer}", answer.answer === "yes" ? copy.answer.yes_label : copy.answer.no_label);
  if (navigator.share) {
    try {
      await navigator.share({ text, url });
      return;
    } catch {
      // user cancelled the native share sheet — fall through to clipboard as a backup.
    }
  }
  await navigator.clipboard.writeText(url);
  const original = button.textContent;
  button.textContent = "Link copied";
  setTimeout(() => {
    button.textContent = original ?? SHARE_LABEL;
  }, 1500);
}

/** Mean |lateral bias| when one antenna's Johnston's-organ neurons are driven (+0.118 left,
 * -0.157 right; docs/MODEL.md §Calibration). Fallback only: the API sends `turn_strength`. */
const SOUND_CUE_BIAS = 0.1375;

export function turnStrength(answer: Pick<Answer, "turn_strength" | "lab">): number {
  const raw = answer.turn_strength ?? Math.abs(answer.lab.corrected_bias) / SOUND_CUE_BIAS;
  return Number.isFinite(raw) ? Math.min(1, Math.max(0, raw)) : 0;
}

export function buildAnswerCard(answer: Answer, opts: { showQuestion?: boolean } = {}): HTMLElement {
  // YES/NO is shown typographically (design-style-reference.md: "no green/red"). No direction
  // arrow any more: the 3D fly turning to its label shows the direction, and a physical-side
  // arrow contradicted it on screen (the fly faces the viewer, so its right is screen-left).
  const verdict = h("div", { class: "verdict" }, answer.answer === "yes" ? copy.answer.yes_label : copy.answer.no_label);

  // Turn strength, not the old 50-100% "confidence": that scale started at 50%, so every real
  // answer read as a medium turn (51-53%). Measured against a real stimulus instead: the turn the
  // same brain makes when one antenna's hearing neurons are driven (calibration, docs/MODEL.md).
  const strengthPct = Math.round(turnStrength(answer) * 100);
  const confidence = h("p", { class: "label-mono" }, `${copy.answer.confidence_label}: ${strengthPct}% ${copy.answer.strength_reference}`);
  const caveat = h("p", { class: "answer-caveat" }, copy.answer.caveat);

  const bars = h("div", { class: "state-bars" });
  for (const key of STATE_KEYS) {
    const value = answer.states[key];
    if (value === null) continue;
    const meta = copy.answer.states[key];
    const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
    const row = h("div", { class: "state-bar", title: meta?.tooltip ?? "" }, [
      h("span", {}, meta?.label ?? key),
      h("span", { class: "track" }, h("span", { class: "fill", style: `width:${pct}%` })),
      h("span", { class: "state-value" }, `${pct}%`),
    ]);
    bars.appendChild(row);
  }

  const shareBtn = h("button", { class: "btn btn-ghost" }, SHARE_LABEL);
  shareBtn.addEventListener("click", () => void share(answer, shareBtn));

  const question = h("p", {}, `“${answer.question}”`);

  // No "real spikes, slowed 10x" tag here (2026-09-21): this card has no raster in it, and every
  // caller — stageView via home.ts and share.ts — already puts that tag on the brain canvas it
  // actually labels. Having it here too printed the same pill twice per answer, the second one
  // orphaned mid-paragraph between the caveat and the meters.
  // No "panel" class here (2026-09-19 layout audit): every caller now embeds this inside its own
  // card (home.ts's `.answer-row`, share.ts's `.answer-row`) — drawing a second bordered/shadowed
  // box around it would nest two cards inside each other.
  return h("div", { class: "answer-card" }, [
    ...(opts.showQuestion === false ? [] : [question]),
    verdict,
    confidence,
    caveat,
    bars,
    h("div", { class: "ask-row" }, [shareBtn]),
  ]);
}
