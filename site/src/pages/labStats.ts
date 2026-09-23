// "Lab stats" section: live counters from GET /api/stats, refreshed on every SSE "answered"
// event and every 30s as a fallback. Every number is fmt*-formatted so a missing/NaN value
// renders "—" instead of a stale or fabricated figure.
import { h, clear } from "../dom";
import { getStats } from "../api/client";
import { fmtNum, fmtPct } from "../format";
import type { Stats } from "../api/types";

const HONESTY_BASE =
  "The fly usually gives the same answer to a question and to its opposite. That's expected: it can't read. It reacts to the pattern the words make, not to their meaning.";

const POLL_MS = 30000;

const STATE_LABELS: Record<keyof Stats["state_counts"], string> = {
  appetite: "Appetite",
  fear: "Startle",
  backoff: "Backed off",
  courtship: "Courtship",
  arousal: "Whole-brain buzz",
};

export interface LabStats {
  el: HTMLElement;
  refresh: () => void;
  destroy: () => void;
}

function statCard(value: string, label: string): HTMLElement {
  return h("div", { class: "stat-card" }, [h("div", { class: "stat-value" }, value), h("div", { class: "label-mono" }, label)]);
}

/** 2,158,468 → "2.2 million": the strip reads at a glance; the exact figure sits in the title. */
export function fmtCompact(n: number): string {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(1)} billion`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)} million`;
  return fmtNum(n);
}

// Compact since 2026-09-19 (the numbers block was too bulky): one strip of four figures,
// everything else one click away under "More numbers".
export function mountLabStats(): LabStats {
  const strip = h("div", { class: "stats-strip" });
  const gridDerived = h("div", { class: "stats-grid stats-grid-3" });
  const stateGrid = h("div", { class: "stats-grid stats-grid-small" });
  const repeatNote = h("p", { class: "helper-text" });
  const errorNote = h("p", { class: "rejection", "aria-live": "polite" });
  const honestyNote = h("p", { class: "helper-text honesty-note" }, HONESTY_BASE);

  const el = h("section", { class: "lab-strip", id: "lab-stats" }, [
    h("h2", { class: "label-mono lab-strip-title" }, "Live numbers"),
    strip,
    h("details", { class: "lab-strip-more" }, [
      h("summary", {}, "More numbers"),
      gridDerived,
      h("h4", {}, "Which behaviour programs have fired"),
      stateGrid,
      repeatNote,
      honestyNote,
    ]),
    errorNote,
  ]);

  function stripItem(value: string, label: string, exact?: string): HTMLElement {
    return h("div", { class: "strip-item", ...(exact ? { title: exact } : {}) }, [
      h("span", { class: "strip-value" }, value),
      h("span", { class: "label-mono" }, label),
    ]);
  }

  function render(stats: Stats) {
    clear(strip);
    strip.appendChild(stripItem(fmtNum(stats.answered), "questions answered"));
    strip.appendChild(stripItem(`${fmtNum(stats.yes)} / ${fmtNum(stats.no)}`, "yes / no"));
    strip.appendChild(stripItem(fmtCompact(stats.total_spikes), "real spikes", fmtNum(stats.total_spikes)));
    strip.appendChild(stripItem(`${fmtNum(stats.simulated_ms / 1000, 1)} s`, "of brain time simulated"));

    clear(gridDerived);
    gridDerived.appendChild(statCard(fmtNum(stats.neuron_updates), "neuron state updates"));
    gridDerived.appendChild(statCard(fmtNum(stats.mean_abs_corrected_bias, 3), "mean |corrected bias|"));
    gridDerived.appendChild(statCard(fmtPct(stats.mean_active_fraction), "mean active fraction"));

    clear(stateGrid);
    (Object.keys(STATE_LABELS) as Array<keyof Stats["state_counts"]>).forEach((key) => {
      stateGrid.appendChild(statCard(fmtNum(stats.state_counts[key]), STATE_LABELS[key]));
    });

    // Identical wording reuses the same random seed (pipeline.question_key), so "asked twice, same
    // answer" is by construction and no longer a measurement worth showing; what stays honest to
    // show is that rewording is a new input and can flip the answer.
    repeatNote.textContent =
      "The same question, worded the same way, always replays the same simulation and gets the same answer. Reword it and the fly gets a different input, so the answer can change.";
    honestyNote.textContent = HONESTY_BASE;
  }

  function refresh() {
    getStats()
      .then((stats) => {
        errorNote.textContent = "";
        render(stats);
      })
      .catch((err) => {
        errorNote.textContent = "Stats are temporarily unavailable.";
        console.error("getStats failed", err);
      });
  }

  refresh();
  const timer = window.setInterval(refresh, POLL_MS);

  return { el, refresh, destroy: () => clearInterval(timer) };
}
