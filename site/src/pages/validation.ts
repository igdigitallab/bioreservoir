// "Validation" section: brain passport (both datasets), the server-computed lateral-validation
// verdict each brain passed (or failed) before it ever answered a live question, handedness
// correction.
// Fetched once from GET /api/validation — this data does not change per question.
//
// Field names below match the REAL backend response (`bioreservoir.live.validation.
// build_validation()`, checked against a live dev-mode call 2026-09-18) — see api/types.ts for
// why this is keyed per dataset instead of a single "passport".
import { h, clear } from "../dom";
import { copy } from "../content";
import { renderHeadline } from "../headline";
import { getValidation } from "../api/client";
import { fmtNum, fmtStr } from "../format";
import type { BrainPassportEntry, LateralValidationEntry, Validation } from "../api/types";

// Static citation facts (docs/DATA.md, footer.attribution in copy.json) — not returned by the
// API, which only carries the numbers it can derive from the harmonized graph itself.
const DATASET_INFO: Record<string, { label: string; license: string; source: string }> = {
  malecns: { label: "MaleCNS v1.0 (male)", license: "CC BY 4.0", source: "male-cns.janelia.org" },
  banc: { label: "BANC (female)", license: "CC BY 4.0", source: "Harvard Dataverse DOI:10.7910/DVN/8TFGGB" },
};

function passportRow(dataset: string, entry: BrainPassportEntry): HTMLElement {
  const info = DATASET_INFO[dataset] ?? { label: dataset, license: "—", source: "—" };
  return h("div", { class: "passport" }, [
    h("h4", {}, info.label),
    h("div", { class: "passport-grid" }, [
      h("div", {}, [h("span", { class: "label-mono" }, "neurons"), h("span", {}, fmtNum(entry.n_neurons))]),
      h("div", {}, [h("span", { class: "label-mono" }, "connections"), h("span", {}, fmtNum(entry.n_connections))]),
      h("div", {}, [h("span", { class: "label-mono" }, "synapses"), h("span", {}, fmtNum(entry.n_synapses))]),
      h("div", {}, [h("span", { class: "label-mono" }, "min synapses/edge"), h("span", {}, fmtNum(entry.min_syn))]),
      h("div", {}, [h("span", { class: "label-mono" }, "license"), h("span", {}, fmtStr(info.license))]),
      h("div", {}, [h("span", { class: "label-mono" }, "source"), h("span", {}, fmtStr(info.source))]),
    ]),
  ]);
}

/** Status badge + plain-English label are always visible; the raw numeric readout (mean ± std
 * per side, the gap, the pass threshold) sits behind a <details> disclosure — same
 * "expanded on desktop, collapsible" pattern as the lab readout panel (labReadout.ts) — instead
 * of always-on monospace lines reading as a terminal dump next to a plain-English status
 * (2026-09-19 layout audit). No numbers are hidden, only deferred one click/tap away. */
function lateralCheckRow(name: string, entry: LateralValidationEntry | null | undefined): HTMLElement {
  if (!entry) {
    return h("div", { class: "check-row" }, [h("span", { class: "check-badge check-unknown" }, "—"), h("span", { class: "check-row-label" }, name)]);
  }
  const badgeClass = entry.passed ? "check-badge check-passed" : "check-badge check-failed";
  const badgeText = entry.passed ? "PASSED" : "FAILED";
  const left = entry.jo_left_driven_descending_bias;
  const right = entry.jo_right_driven_descending_bias;
  return h("div", { class: "check-row" }, [
    h("span", { class: badgeClass }, badgeText),
    h("div", {}, [
      h("div", { class: "check-row-label" }, `${name}: descending_all lateral bias, JO-left vs JO-right drive`),
      h("details", { class: "check-detail" }, [
        h("summary", {}, "Show the numbers"),
        h("div", { class: "check-detail-body" }, [
          h(
            "div",
            { class: "label-mono" },
            `${fmtNum(left.mean, 3)} ± ${fmtNum(left.std, 3)} (n=${left.n}) / ${fmtNum(right.mean, 3)} ± ${fmtNum(right.std, 3)} (n=${right.n})`,
          ),
          h("div", { class: "helper-text" }, `left/right gap: ${fmtNum(entry.left_vs_right_bias_gap, 3)} (pass threshold: opposite-signed AND gap > 0.05)`),
        ]),
      ]),
    ]),
  ]);
}

export function mountValidation(): { el: HTMLElement; destroy: () => void } {
  const body = h("div", { class: "validation-body" }, h("p", {}, "Loading validation data…"));
  const el = h("section", { class: "panel validation", id: "validation" }, [h("h2", {}, renderHeadline(copy.sections.validation)), body]);

  getValidation()
    .then((v: Validation) => {
      clear(body);
      body.appendChild(h("p", { class: "helper-text" }, fmtStr(v.model)));
      for (const dataset of Object.keys(v.brain_passport)) {
        body.appendChild(passportRow(dataset, v.brain_passport[dataset]!));
      }
      body.appendChild(
        h("div", { class: "calibration" }, [
          h("h4", {}, "Checks passed before this brain ever answered a live question"),
          lateralCheckRow("Lateral validation — male (MaleCNS)", v.lateral_validation.malecns),
          lateralCheckRow("Lateral validation — female (BANC)", v.lateral_validation.banc),
        ]),
      );
      const hb = v.handedness_b0;
      body.appendChild(
        h("div", { class: "handedness" }, [
          h("h4", {}, "Handedness correction"),
          h(
            "p",
            {},
            `${fmtNum(hb.n_reference_sentences)} reference sentences (${hb.brain}, condition=${hb.condition}) · b0 = ${
              hb.b0 === null ? "— (not enough reference runs yet)" : fmtNum(hb.b0, 4)
            }`,
          ),
        ]),
      );
    })
    .catch((err) => {
      clear(body);
      body.appendChild(h("p", { class: "rejection" }, "Validation data is temporarily unavailable."));
      console.error("getValidation failed", err);
    });

  return { el, destroy: () => undefined };
}
