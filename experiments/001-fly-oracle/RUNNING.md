# Running experiment 001 (Fly Oracle)

All commands via `python -m bioreservoir.oracle <run|score> [--config config.yaml]`. Every seed and
parameter lives in [`config.yaml`](config.yaml) — values marked `TO_CALIBRATE` there are
placeholders until `sim/calibrate.py`'s report finalizes them (docs/MODEL.md "Calibration").

## Dry run (no simulation, no cage needed)

```bash
~/.local/bin/uv run --no-sync python -m bioreservoir.oracle run --dry-run
```

Prints the full work plan (questions x brains x {real, rewired, er} x {original, negation,
paraphrase}, plus the no-brain and coin controls) **and** the handedness reference batch (24
sentences in [`reference.yaml`](reference.yaml) x brains x {real, rewired, er, no_brain} — see
README.md "Handedness and side mapping"), each with its own CPU-hour estimate from
`sim.calibrate.throughput`, seeded with docs/MODEL.md's own measured s-wall/s-simulated figures
(MaleCNS 27.48, BANC 18.98). For the current 33-question set, 24 reference sentences and
`config.yaml`'s `n_trials: 10` / `duration_ms: 250.0`:

| | sim trials | sequential CPU-hours | `--workers 3` hours |
|---|---|---|---|
| questions (`cost_estimate`) | 5,940 | 9.58 | 3.19 |
| reference batch (`reference_cost_estimate`) | 1,440 | 2.32 | 0.77 |
| combined (`combined_cost_estimate`) | 7,380 | 11.91 | 3.97 |

Re-run this after any change to `config.yaml`'s `trial:` block, or to `reference.yaml`'s sentence
count — neither estimate is cached.

## Run (writes to the ledger, resumable)

```bash
CAGE='systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 -p CPUQuota=400% nice -n 19'
export XDG_RUNTIME_DIR=/run/user/$(id -u)
$CAGE ~/.local/bin/uv run --no-sync python -m bioreservoir.oracle run --workers 3
```

Skips every `(question, brain, condition, variant)` the ledger already has `status='done'` under
the *current* `config.yaml` hash (`--no-resume` forces a full rerun). One `LIFNetwork` is built per
`(brain, condition)` — 6 groups (2 brains x {real, rewired, er}) — reused via store/restore across
every question/variant/trial **and** that group's own 24 handedness reference sentences (run
first, so the group's `b0` is defined before any question in it — README.md "Handedness and side
mapping"); `--workers N` runs up to `N` groups in parallel. The no-brain/coin controls run directly
in the calling process (no connectome, effectively free) — the no-brain control also runs its own
24 reference sentences first, per brain. `--question ID` (repeatable) restricts which *questions*
run; the reference batch always runs in full for every `(brain, condition)`, independent of
`--question`, since it is not itself a question. New `done` predictions are exported to
`predictions/<UTC-date>.jsonl` and `predictions/SHA256SUMS` is refreshed — `git add` and commit
those (this command does not commit). Reference-run results live only in the ledger's
`reference_runs`/`reference_predictions` tables — they are never scored and are not exported to
`predictions/*.jsonl`.

`--stamp` also runs `ots stamp predictions/SHA256SUMS` (opentimestamps-client, the `stamp` extra:
`uv sync --extra stamp`) after export. Implemented but never invoked in CI or by any default
command — it is run manually, on demand, before a batch's predictions are disclosed.

## Score

```bash
~/.local/bin/uv run --no-sync python -m bioreservoir.oracle score
```

Reads every `done` prediction with a recorded resolution (`Ledger.record_resolution`, not yet
wired to an automatic feed — resolutions are entered by hand from `resolution_source` once a race
is called), computes accuracy/Brier/log-loss/calibration per `(brain, condition)` on the
pre-registered `original` text variant, a "clustered" view (questions grouped by `category` —
every 2026 midterm question shares `category: politics`, so they count as one cluster, not 31
independent draws), and the negation/paraphrase invariance report. Writes
`site/data/scoreboard.json` and prints the same report as JSON. `--stamp` re-stamps
`predictions/SHA256SUMS` after scoring (same caveat as `run --stamp`).

## Where outputs go

| Path | What | In git? |
|---|---|---|
| `experiments/001-fly-oracle/ledger.sqlite` | working state (runs, predictions, resolutions) | no (`*.sqlite`) |
| `experiments/001-fly-oracle/predictions/<date>.jsonl` | append-only export, one file per day | **yes** |
| `experiments/001-fly-oracle/predictions/SHA256SUMS` | hash of every `.jsonl` above | **yes** |
| `data/processed/<dataset>-min5-{rewired,er}-seed<N>/` | cached control graphs | no (`data/`) |
| `site/data/scoreboard.json` | scoring report for the future site | not yet decided — see open issues |

## Resuming after an interrupted run

Just re-run the same `run --workers N` command with the same `config.yaml`: anything already
`done` under that config's hash is skipped; anything left `running` (a worker died mid-batch) or
`failed` is retried from scratch — `LIFNetwork` has no partial-trial checkpoint.

## Open issues (for the calibration/lead review)

- `config.yaml`'s `trial`/`input`/`readout` blocks are `TO_CALIBRATE` placeholders — `readout.name:
  descending_all` is docs/MODEL.md's own calibration-tested candidate (DNa01/DNa02 measured 0
  spikes at these settings); `input.population`/rate range are carried over from
  `sim/calibrate.py`'s `input_regime_scan` grid, not yet a calibration decision.
- No automatic resolution feed — `Ledger.record_resolution` must be called by hand once AP calls
  each race; nothing here fetches results itself (platform/market APIs are off-limits).
