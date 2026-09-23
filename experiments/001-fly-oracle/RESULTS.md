# Batch 001 — process data

Full per-run measurement table for the batch referenced from the site's `/data` page and from
`docs/DATA.md`: 33 pre-registered yes/no questions about the 2026 US midterms, each run on the
real male connectome (MaleCNS v1.0) and four controls. Every row below comes straight out of
`ledger.sqlite`'s `runs`/`predictions` tables — nothing here is hand-edited.

**Full per-run values: [`results.csv`](results.csv) — 429 rows, one per run.**
Columns: `run_id, question_id, brain, condition, variant, p_yes, mean_bias, corrected_bias, b0,
n_trials, n_zero_spike_trials, left_is_yes`.

## Volume by condition

| Condition | Runs | What it is |
|---|---|---|
| `real` | 99 | Real male connectome (33 questions × 3 phrasings: original, negation, paraphrase), 10 simulation trials per run |
| `rewired` | 99 | Same neurons, degrees, synaptic weights, signs and cell types as `real`; connections shuffled (degree-preserving rewire), 10 trials per run |
| `er` | 99 | Erdős–Rényi random graph, same size and density as the real connectome, 10 trials per run |
| `no_brain` | 99 | Same text-embedding and readout wiring as `real`, no connectome at all, 1 deterministic trial per run |
| `coin` | 33 | Seeded pseudo-random draw, one per question, 1 deterministic trial per run |
| **Total** | **429** | |

`real`, `rewired` and `er` are stochastic simulations (10 trials each); `no_brain` and `coin` are
deterministic given their seed (1 trial each) — that is why 429 does not split evenly across five
conditions.

## Real-brain invariance (condition=`real`, 33 questions × 3 phrasings)

Computed from the 99 real-connectome runs only, comparing each question's original phrasing
against its negation and its paraphrase.

| Measure | Value |
|---|---|
| Questions | 33 |
| Mean p(yes), original phrasing | 0.5001 |
| Range of p(yes) across questions | 0.4930 – 0.5053 |
| Mean gap: question vs. its negation's complement | 0.0038 |
| Mean gap: question vs. its paraphrase | 0.0019 |
| Largest single negation gap | 0.0136 |
| Largest single paraphrase gap | 0.0061 |
| Zero-spike trials | 0 of 990 (the brain fired on every trial) |

Per-question, per-condition values for `rewired`, `er`, `no_brain` and `coin` are in
[`results.csv`](results.csv) alongside `real` — this table only aggregates the `real` rows because
that is the condition the invariance claim (negation/paraphrase) is about.

## What this batch does not yet contain

These 33 questions resolve by 2026-12-15. No outcomes have been recorded in the ledger's
`resolutions` table yet, so no accuracy, Brier score or calibration-against-outcome numbers exist
for this batch — only the process numbers above. That scoring will be added here once the
questions resolve, whichever way the brain (and each control) landed.

## Related

- Pre-registration: [`stamps/`](stamps/) — OpenTimestamps proofs, committed before this batch ran.
- Sealed run-level predictions for the midterms themselves:
  [`predictions/2026-09-19.jsonl`](predictions/2026-09-19.jsonl), released at resolution (after
  2026-11-03); its hash is in [`predictions/SHA256SUMS`](predictions/SHA256SUMS) and is timestamped
  now.
- Model, parameters and biological validation: [`../../docs/MODEL.md`](../../docs/MODEL.md).
- Data sources and licenses: [`../../docs/DATA.md`](../../docs/DATA.md).
- Methodology narrative: [`README.md`](README.md).
