// Per-answer "Lab readout" panel: the actual decision math, a real spike raster, the named cell
// types that fired, scale numbers, and a reproduce command — the "prove this is a real fly
// brain, not a game" panel. Every number here is read straight off Answer.lab; format.ts's
// fmt* helpers render "—" for anything null/missing rather than inventing a number.
// <details>/<summary> gives the "expanded on desktop, collapsible on mobile" behaviour for
// free (native, no extra JS) — CSS just widens the default triangle marker on small screens.
import { h } from "../dom";
import { buildDecisionMath } from "../raster";
import { buildRasterView } from "./rasterView";
import { fmtMs, fmtNum, fmtPct, fmtSigned, fmtStr } from "../format";
import type { Answer } from "../api/types";

function copyButton(text: string): HTMLButtonElement {
  const btn = h("button", { class: "btn btn-ghost btn-copy", type: "button" }, "Copy") as HTMLButtonElement;
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      const original = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = original ?? "Copy"), 1500);
    } catch {
      // Clipboard API can be blocked (permissions, insecure context) — fail silently, the
      // command is still visible and selectable by hand right next to the button.
    }
  });
  return btn;
}

function decisionMathBlock(answer: Answer): HTMLElement {
  const lab = answer.lab;
  const math = buildDecisionMath(lab);

  const trialRows = lab.trials.map((t, i) =>
    h("tr", {}, [
      h("td", {}, fmtNum(t.seed)),
      h("td", {}, `${fmtNum(t.spikes_left)} L`),
      h("td", {}, `${fmtNum(t.spikes_right)} R`),
      h("td", {}, fmtSigned(t.bias)),
      h("td", { class: "label-mono" }, `trial ${i + 1}`),
    ]),
  );

  const trialsTable = h("div", { class: "lab-table-wrap" }, h("table", { class: "lab-table" }, [
    h("thead", {}, h("tr", {}, [h("th", {}, "seed"), h("th", {}, "L spikes"), h("th", {}, "R spikes"), h("th", {}, "raw bias"), h("th", {}, "")])),
    h("tbody", {}, trialRows),
  ]));

  const chain = h("div", { class: "decision-chain" }, [
    h("div", { class: "decision-step" }, [h("span", { class: "label-mono" }, "mean raw bias"), h("span", {}, fmtSigned(math.meanRawBias))]),
    h("div", { class: "decision-step" }, [h("span", { class: "label-mono" }, "minus handedness b0"), h("span", {}, fmtSigned(math.b0))]),
    h("div", { class: "decision-step decision-step-result" }, [
      h("span", { class: "label-mono" }, "corrected bias"),
      h("span", {}, fmtSigned(math.correctedBias)),
    ]),
    h("div", { class: "decision-step decision-step-result" }, [
      h("span", { class: "label-mono" }, "yes side"),
      h("span", {}, fmtStr(math.yesSide)),
    ]),
  ]);

  return h("div", { class: "decision-math" }, [
    trialsTable,
    chain,
    h(
      "p",
      { class: "helper-text" },
      "b0 is the mean lateral bias over 24 neutral reference sentences for this brain — subtracted so the brain's own left/right wiring bias never leaks into the answer.",
    ),
  ]);
}

function topCellTypesTable(answer: Answer): HTMLElement {
  const rows = answer.lab.top_cell_types.map((ct) =>
    h("tr", {}, [
      h("td", {}, fmtStr(ct.cell_type)),
      h("td", { class: "label-mono" }, fmtStr(ct.super_class)),
      h("td", {}, fmtNum(ct.n_neurons)),
      h("td", {}, fmtNum(ct.spikes)),
      h("td", {}, `${fmtNum(ct.rate_hz, 1)} Hz`),
    ]),
  );
  if (rows.length === 0) {
    rows.push(h("tr", {}, h("td", { colSpan: "5" }, "—")));
  }
  return h("div", { class: "lab-table-wrap" }, h("table", { class: "lab-table" }, [
    h(
      "thead",
      {},
      h("tr", {}, [h("th", {}, "cell type"), h("th", {}, "class"), h("th", {}, "n"), h("th", {}, "spikes"), h("th", {}, "rate")]),
    ),
    h("tbody", {}, rows),
  ]));
}

function scaleNumbers(answer: Answer): HTMLElement {
  const lab = answer.lab;
  return h("div", { class: "scale-numbers" }, [
    h("div", {}, [h("span", { class: "scale-value" }, fmtNum(lab.active_neurons)), h("span", { class: "label-mono" }, ` of ${fmtNum(lab.provenance.n_neurons)} neurons active (${fmtPct(lab.active_fraction)})`)]),
    h("div", {}, [h("span", { class: "scale-value" }, fmtNum(lab.total_spikes)), h("span", { class: "label-mono" }, " total spikes")]),
    h("div", {}, [h("span", { class: "scale-value" }, fmtNum(lab.stimulated.total)), h("span", { class: "label-mono" }, ` sensory neurons stimulated (${fmtNum(lab.stimulated.left)} L / ${fmtNum(lab.stimulated.right)} R)`)]),
    h("div", {}, [h("span", { class: "scale-value" }, fmtMs(lab.readout_latency_ms)), h("span", { class: "label-mono" }, " readout latency")]),
  ]);
}

function reproduceBlock(answer: Answer): HTMLElement {
  const p = answer.lab.provenance;
  const stamp = h(
    "p",
    { class: "label-mono" },
    `code ${fmtStr(p.code_sha)} · config ${fmtStr(p.config_hash)} · ${fmtStr(p.model)} · dt=${fmtNum(p.dt_ms, 2)}ms · min_syn=${fmtNum(p.min_syn)}`,
  );
  // No command while the repository is private (provenance.reproduce === null): the hashes below
  // still pin exactly which code and config produced this answer, and saying so is honest, where
  // a `git clone` of a 404 repo was not.
  if (!p.reproduce) {
    return h("div", { class: "reproduce-block" }, [
      h("p", { class: "helper-text" }, "The code is not published yet. These hashes pin the exact version that produced this answer, and the command to re-run it appears here when the repository goes public."),
      stamp,
    ]);
  }
  return h("div", { class: "reproduce-block" }, [
    h("pre", { class: "reproduce-cmd" }, p.reproduce),
    copyButton(p.reproduce),
    stamp,
  ]);
}

export function buildLabReadout(answer: Answer): HTMLElement {
  const lab = answer.lab;
  const raster = buildRasterView(lab, { width: 640, interactive: true });

  const details = h("details", { class: "panel lab-readout" }, [
    h("summary", {}, "Lab readout — how the brain decided"),
    h("div", { class: "lab-readout-body" }, [
      h("h4", {}, "Decision math"),
      decisionMathBlock(answer),
      h("h4", {}, "Spike raster (trial 1, real spikes)"),
      raster.el,
      h("h4", {}, "Top cell types that fired"),
      topCellTypesTable(answer),
      h("h4", {}, "Scale"),
      scaleNumbers(answer),
      h("h4", {}, "Recompute this answer"),
      reproduceBlock(answer),
    ]),
  ]);
  // Expanded by default on desktop, collapsed by default on mobile — <details> is natively
  // collapsible either way, this only sets the initial state per the build brief.
  details.open = window.matchMedia("(min-width: 768px)").matches;
  return details;
}
