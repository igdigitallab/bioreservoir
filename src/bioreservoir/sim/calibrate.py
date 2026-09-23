"""Calibration runs for the ported Shiu et al. LIF model (see `sim/lif.py`), on both real
connectomes. Entry point:

    python -m bioreservoir.sim.calibrate <dataset> [--check NAME ...] [options]

Must be run inside the memory/CPU cage described in experiments/001-fly-oracle/RUNNING.md (same convention as
`sim/bench.py`, which this module reuses for graph loading and population lookup).

Every check below drives only named BIOLOGICAL sensory populations (gustatory, Johnston's
organ, generic "sensory_all") at fixed Poisson rates — never question text, never the sentence
encoder. This is a hard rule (see the task brief this module was written against): calibration
against election questions would be p-hacking. Every trial is seeded, so every number here is
exactly reproducible with `--seed`.

One `LIFNetwork.run_trial` call returns spike counts for **every** neuron (`monitor_idx=None` in
`bench.graph_to_arrays`'s callers), so a single simulated trial under one stimulus can be read
out against many different named readout populations without re-simulating — only the *stimulus*
(which population(s) are driven, at what rate, for how long) needs a fresh trial. `run_condition`
below is built around that: one simulation per (driven-population-set, rate, duration) triple,
readouts computed afterwards from the same spike-count array.

Checks (see each function's docstring for what it measures and why):
    dose-response     sugar GRN -> MN9 firing rate at 100/150/200 Hz (Shiu's own protocol).
    bitter-suppression sugar-only vs sugar+bitter co-drive -> MN9 (Shiu's other headline result).
    lateral-bias       JO left/right -> descending_all_{left,right} population bias, the
                        candidate primary cross-brain readout (see docs/MODEL.md).
    approach-escape    MN9/feeding vs DNp01+MDN/escape under appetitive vs aversive drive.
    input-regime       generic sensory_all random-subset x rate grid, looking for a regime
                        where the primary readout responds without saturating.
    duration-scan       sugar GRN -> MN9 at shorter trial durations, to find the shortest
                        duration that keeps the dose-response effect.
    throughput          wall time per trial for the recommended settings, extrapolated to the
                        full election-question batch (see docs/MODEL.md "Throughput").

Output: a JSON report to stdout (or --out FILE), one top-level key per check that ran.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from bioreservoir.sim import bench
from bioreservoir.sim.lif import LIFNetwork

DEFAULT_TRIAL_MS = 1000.0
DEFAULT_MIN_SYN = 5

# Readout populations checked by every stimulus-driven condition below (see populations.yaml).
# Populations absent for a dataset (e.g. MN9_proboscis in BANC) are reported as `null`, not
# skipped silently, so an absence is visible in the JSON output itself.
CORE_READOUTS = (
    "MN9_proboscis",
    "DNp01_giant_fiber",
    "MDN",
    "DNa01_left",
    "DNa01_right",
    "DNa02_left",
    "DNa02_right",
    "descending_all_left",
    "descending_all_right",
)


@dataclass(frozen=True)
class Built:
    dataset: str
    net: LIFNetwork
    id_to_dense: dict[int, int]


def build(dataset: str, min_syn: int = DEFAULT_MIN_SYN, codegen_target: str | None = None) -> Built:
    n_neurons, pre_idx, post_idx, weight, id_to_dense = bench.graph_to_arrays(dataset, min_syn=min_syn)
    net = LIFNetwork(n_neurons, pre_idx, post_idx, weight, codegen_target=codegen_target)
    return Built(dataset=dataset, net=net, id_to_dense=id_to_dense)


def _readout_sums(built: Built, spike_counts: np.ndarray, names: tuple[str, ...]) -> dict[str, int | None]:
    """Sum `spike_counts` over each named population; `None` if the population is absent."""
    out: dict[str, int | None] = {}
    for name in names:
        idx = bench.population_indices(built.dataset, built.id_to_dense, name)
        out[name] = None if idx is None else int(spike_counts[idx].sum()) if idx.size else 0
    return out


def run_condition(
    built: Built,
    driven: dict[str, float],
    n_trials: int,
    trial_ms: float,
    base_seed: int,
    readouts: tuple[str, ...] = CORE_READOUTS,
) -> dict:
    """Run `n_trials` independent trials with the population(s) in `driven` ({name: rate_hz})
    Poisson-stimulated together, and report per-trial readout spike sums plus whole-network
    activity stats (total spikes, fraction of neurons active) to catch runaway activity.
    """
    driven_idx: list[np.ndarray] = []
    driven_rate: list[np.ndarray] = []
    driven_counts: dict[str, int] = {}
    for name, rate in driven.items():
        idx = bench.population_indices(built.dataset, built.id_to_dense, name)
        if idx is None or idx.size == 0:
            raise ValueError(f"driven population {name!r} absent or empty for {built.dataset!r}")
        driven_idx.append(idx)
        driven_rate.append(np.full(len(idx), rate))
        driven_counts[name] = len(idx)
    all_idx = np.concatenate(driven_idx)
    all_rate = np.concatenate(driven_rate)

    trials = []
    for t in range(n_trials):
        result = built.net.run_trial((all_idx, all_rate), duration_ms=trial_ms, seed=base_seed + t)
        row: dict = _readout_sums(built, result.spike_counts, readouts)
        row["total_spikes"] = int(result.spike_counts.sum())
        row["n_active_neurons"] = int((result.spike_counts > 0).sum())
        row["wall_time_s"] = result.wall_time_s
        trials.append(row)

    return {
        "driven_rate_hz": driven,
        "driven_n_neurons": driven_counts,
        "trial_ms": trial_ms,
        "n_trials": n_trials,
        "n_total_neurons": built.net.n_neurons,
        "trials": trials,
    }


def _mean_std(trials: list[dict], key: str) -> dict[str, float | None]:
    values = [t[key] for t in trials if t.get(key) is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def dose_response(
    built: Built,
    rates_hz: tuple[float, ...] = (100.0, 150.0, 200.0),
    n_trials: int = 10,
    trial_ms: float = DEFAULT_TRIAL_MS,
    seed: int = 1000,
) -> dict:
    """Shiu's own protocol: sugar GRNs (`gustatory_sugar`, cross-dataset LB3a/LB3b match, see
    populations.yaml) at 100/150/200 Hz for 1 s -> MN9 firing rate. MN9 is MaleCNS-only (absent
    in BANC, populations.yaml); BANC still runs the same stimulus and reports whole-network
    activity + the candidate cross-brain readout (`descending_all_*`) so the check is not
    silently skipped, just without the MN9 number Shiu's own headline result used.
    """
    out = {}
    for i, rate in enumerate(rates_hz):
        cond = run_condition(built, {"gustatory_sugar": rate}, n_trials, trial_ms, seed + i * 10_000)
        cond["mn9_rate_hz"] = _mean_std(
            [{"v": t["MN9_proboscis"] / (trial_ms / 1000.0) if t["MN9_proboscis"] is not None else None} for t in cond["trials"]],
            "v",
        )
        cond["fraction_active"] = _mean_std(
            [{"v": t["n_active_neurons"] / cond["n_total_neurons"]} for t in cond["trials"]], "v"
        )
        out[f"{rate:.0f}hz"] = cond
    return out


def bitter_suppression(
    built: Built, rate_hz: float = 150.0, n_trials: int = 10, trial_ms: float = DEFAULT_TRIAL_MS, seed: int = 2000
) -> dict:
    """Shiu's other headline gustatory result: bitter GRNs co-activated with sugar GRNs suppress
    MN9 firing relative to sugar alone (feed-forward inhibition). `gustatory_bitter` now exists
    for both datasets via the same cross-dataset LB1a/LB1b match as gustatory_sugar (task 3).
    """
    sugar_only = run_condition(built, {"gustatory_sugar": rate_hz}, n_trials, trial_ms, seed)
    sugar_bitter = run_condition(
        built, {"gustatory_sugar": rate_hz, "gustatory_bitter": rate_hz}, n_trials, trial_ms, seed + 10_000
    )
    mn9_sugar = _mean_std([{"v": t["MN9_proboscis"]} for t in sugar_only["trials"] if t["MN9_proboscis"] is not None], "v")
    mn9_both = _mean_std(
        [{"v": t["MN9_proboscis"]} for t in sugar_bitter["trials"] if t["MN9_proboscis"] is not None], "v"
    )
    return {
        "sugar_only": sugar_only,
        "sugar_plus_bitter": sugar_bitter,
        "mn9_spikes_sugar_only": mn9_sugar,
        "mn9_spikes_sugar_plus_bitter": mn9_both,
        "suppressed": (
            mn9_both["mean"] < mn9_sugar["mean"] if mn9_sugar["mean"] is not None and mn9_both["mean"] is not None else None
        ),
    }


def _lateral_bias(left_sum: int, right_sum: int) -> float | None:
    total = left_sum + right_sum
    return None if total == 0 else (left_sum - right_sum) / total


def lateral_bias(
    built: Built, rate_hz: float = 150.0, n_trials: int = 5, trial_ms: float = DEFAULT_TRIAL_MS, seed: int = 3000
) -> dict:
    """Candidate primary cross-brain readout: `descending_all_left`/`descending_all_right`
    (n~650/side in both brains, populations.yaml) lateral bias `(L-R)/(L+R)`, driven by
    lateralised Johnston's-organ input (equal neuron counts per side,
    `bench.balanced_lateral_population`). Also reports the smaller DNa01/DNa02 pair for
    comparison (README.md's original "alternative readout").
    """
    left_idx, right_idx, side_counts = bench.balanced_lateral_population(
        built.dataset, built.id_to_dense, bench.LATERAL_INPUT_POPULATION, seed=seed
    )
    out: dict = {"jo_side_counts": side_counts}
    for label, idx in (("jo_left_driven", left_idx), ("jo_right_driven", right_idx)):
        driven_idx = idx
        driven_rate = np.full(len(idx), rate_hz)
        trials = []
        for t in range(n_trials):
            result = built.net.run_trial(
                (driven_idx, driven_rate), duration_ms=trial_ms, seed=seed + 100 + t + (0 if label == "jo_left_driven" else 1000)
            )
            row = _readout_sums(built, result.spike_counts, CORE_READOUTS)
            row["total_spikes"] = int(result.spike_counts.sum())
            row["n_active_neurons"] = int((result.spike_counts > 0).sum())
            desc_l, desc_r = row["descending_all_left"], row["descending_all_right"]
            row["descending_bias"] = _lateral_bias(desc_l, desc_r) if desc_l is not None and desc_r is not None else None
            dna_l = (row["DNa01_left"] or 0) + (row["DNa02_left"] or 0)
            dna_r = (row["DNa01_right"] or 0) + (row["DNa02_right"] or 0)
            row["dna_bias"] = _lateral_bias(dna_l, dna_r)
            trials.append(row)
        out[label] = {
            "n_driven": len(idx),
            "trial_ms": trial_ms,
            "trials": trials,
            "descending_bias": _mean_std([{"v": r["descending_bias"]} for r in trials if r["descending_bias"] is not None], "v"),
            "dna_bias": _mean_std([{"v": r["dna_bias"]} for r in trials if r["dna_bias"] is not None], "v"),
        }
    left_bias = out["jo_left_driven"]["descending_bias"]["mean"]
    right_bias = out["jo_right_driven"]["descending_bias"]["mean"]
    out["left_vs_right_bias_gap"] = (
        None if left_bias is None or right_bias is None else left_bias - right_bias
    )
    return out


def approach_escape(
    built: Built, rate_hz: float = 150.0, n_trials: int = 5, trial_ms: float = DEFAULT_TRIAL_MS, seed: int = 4000
) -> dict:
    """Appetitive (sugar) vs aversive (bitter) drive -> MN9/feeding ("approach") vs
    DNp01+MDN/escape contrast. MN9 is MaleCNS-only; BANC reports the escape side only (DNp01+MDN
    response to appetitive vs aversive stimuli), consistent with MN9_proboscis's documented
    absence in populations.yaml.
    """
    appetitive = run_condition(built, {"gustatory_sugar": rate_hz}, n_trials, trial_ms, seed)
    aversive = run_condition(built, {"gustatory_bitter": rate_hz}, n_trials, trial_ms, seed + 10_000)

    def escape_sum(row: dict) -> int:
        return (row["DNp01_giant_fiber"] or 0) + (row["MDN"] or 0)

    out = {
        "appetitive": appetitive,
        "aversive": aversive,
        "mn9_appetitive": _mean_std([{"v": t["MN9_proboscis"]} for t in appetitive["trials"] if t["MN9_proboscis"] is not None], "v"),
        "mn9_aversive": _mean_std([{"v": t["MN9_proboscis"]} for t in aversive["trials"] if t["MN9_proboscis"] is not None], "v"),
        "escape_appetitive": _mean_std([{"v": escape_sum(t)} for t in appetitive["trials"]], "v"),
        "escape_aversive": _mean_std([{"v": escape_sum(t)} for t in aversive["trials"]], "v"),
    }
    return out


def input_regime_scan(
    built: Built,
    n_driven_values: tuple[int, ...] = (100, 1000),
    rate_values: tuple[float, ...] = (50.0, 150.0, 300.0),
    n_trials: int = 3,
    trial_ms: float = DEFAULT_TRIAL_MS,
    seed: int = 5000,
) -> dict:
    """Generic distributed input: random subsets of `sensory_all` (all primary sensory neurons
    pooled, no modality preference, populations.yaml), scanned over subset size x rate, looking
    for a regime where `descending_all` (the candidate primary readout) responds above baseline
    but total network activity does not saturate (`fraction_active` should stay well under 1.0).
    """
    pool = bench.population_indices(built.dataset, built.id_to_dense, "sensory_all")
    if pool is None:
        raise ValueError(f"sensory_all absent for {built.dataset!r}")
    rng = np.random.default_rng(seed)

    out: dict = {"pool_size": len(pool), "grid": {}}
    for grid_i, n_driven in enumerate(n_driven_values):
        n = min(n_driven, len(pool))
        for grid_j, rate in enumerate(rate_values):
            key = f"n{n_driven}_r{rate:.0f}hz"
            # Deterministic per-grid-point seed offset — NOT `hash(key)`, which is randomized
            # per-process (PYTHONHASHSEED) unless explicitly fixed and would silently break the
            # "reproducible with --seed" guarantee this module's docstring promises.
            grid_seed = seed + (grid_i * len(rate_values) + grid_j) * 1000
            driven_idx = rng.choice(pool, size=n, replace=False)
            driven_rate = np.full(n, rate)
            trials = []
            for t in range(n_trials):
                result = built.net.run_trial(
                    (driven_idx, driven_rate), duration_ms=trial_ms, seed=grid_seed + t
                )
                desc_l = int(result.spike_counts[bench.population_indices(built.dataset, built.id_to_dense, "descending_all_left")].sum())
                desc_r = int(result.spike_counts[bench.population_indices(built.dataset, built.id_to_dense, "descending_all_right")].sum())
                trials.append(
                    {
                        "descending_total": desc_l + desc_r,
                        "descending_bias": _lateral_bias(desc_l, desc_r),
                        "total_spikes": int(result.spike_counts.sum()),
                        "n_active_neurons": int((result.spike_counts > 0).sum()),
                        "wall_time_s": result.wall_time_s,
                    }
                )
            out["grid"][key] = {
                "n_driven": int(n),
                "rate_hz": rate,
                "descending_total": _mean_std([{"v": r["descending_total"]} for r in trials], "v"),
                "fraction_active": _mean_std([{"v": r["n_active_neurons"] / built.net.n_neurons} for r in trials], "v"),
                "trials": trials,
            }
    return out


def duration_scan(
    built: Built,
    durations_ms: tuple[float, ...] = (250.0, 500.0, 1000.0),
    rate_hz: float = 150.0,
    n_trials: int = 5,
    seed: int = 6000,
) -> dict:
    """Shortest trial duration that keeps the sugar-GRN -> MN9 dose-response effect above
    baseline, to size the election-question batch (`throughput`) — a shorter trial that still
    shows the effect directly cuts wall time per trial.
    """
    out = {}
    for duration in durations_ms:
        cond = run_condition(built, {"gustatory_sugar": rate_hz}, n_trials, duration, seed + int(duration))
        cond["mn9_rate_hz"] = _mean_std(
            [{"v": t["MN9_proboscis"] / (duration / 1000.0) if t["MN9_proboscis"] is not None else None} for t in cond["trials"]],
            "v",
        )
        out[f"{duration:.0f}ms"] = cond
    return out


def throughput(
    per_trial_wall_s: dict[str, float],
    n_brains: int = 2,
    n_graph_variants: int = 3,
    n_text_variants: int = 3,
    n_trials_per_condition: int = 10,
    n_questions: int = 33,
    n_parallel: int = 3,
) -> dict:
    """Extrapolate wall-clock hours for the full election batch from measured per-trial times.

    One question = n_brains x n_graph_variants (real, degree-preserving rewired, Erdos-Renyi) x
    n_text_variants (original, negation, paraphrase) x n_trials_per_condition trials. Trials
    split evenly between the two brains' own per-trial wall times; up to `n_parallel` trials run
    concurrently in one cage (up to 3 processes, ~1.6 GB each, well under the 6 GB budget).
    """
    mean_wall_s = float(np.mean(list(per_trial_wall_s.values())))
    trials_per_question = n_brains * n_graph_variants * n_text_variants * n_trials_per_condition
    total_trials = trials_per_question * n_questions
    sequential_hours = total_trials * mean_wall_s / 3600.0
    parallel_hours = sequential_hours / n_parallel
    return {
        "per_trial_wall_s": per_trial_wall_s,
        "mean_wall_s_per_trial": mean_wall_s,
        "trials_per_question": trials_per_question,
        "n_questions": n_questions,
        "total_trials": total_trials,
        "n_parallel": n_parallel,
        "sequential_hours": sequential_hours,
        "parallel_hours": parallel_hours,
    }


CHECKS = (
    "dose-response",
    "bitter-suppression",
    "lateral-bias",
    "approach-escape",
    "input-regime",
    "duration-scan",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["malecns", "banc"])
    parser.add_argument("--check", action="append", choices=CHECKS, help="repeatable; default: all")
    parser.add_argument("--min-syn", type=int, default=DEFAULT_MIN_SYN)
    parser.add_argument("--n-trials", type=int, default=10, help="trials for dose-response/bitter-suppression")
    parser.add_argument("--scan-trials", type=int, default=3, help="trials per grid point for input-regime")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="write JSON here instead of stdout")
    args = parser.parse_args()

    checks = args.check or list(CHECKS)
    t0 = time.monotonic()
    built = build(args.dataset, min_syn=args.min_syn)
    report: dict = {
        "dataset": args.dataset,
        "n_neurons": built.net.n_neurons,
        "n_synapses": built.net.n_synapses,
        "build_time_s": built.net.build_time_s,
    }

    if "dose-response" in checks:
        report["dose_response"] = dose_response(built, n_trials=args.n_trials, seed=args.seed + 1000)
    if "bitter-suppression" in checks:
        report["bitter_suppression"] = bitter_suppression(built, n_trials=args.n_trials, seed=args.seed + 2000)
    if "lateral-bias" in checks:
        report["lateral_bias"] = lateral_bias(built, seed=args.seed + 3000)
    if "approach-escape" in checks:
        report["approach_escape"] = approach_escape(built, seed=args.seed + 4000)
    if "input-regime" in checks:
        report["input_regime"] = input_regime_scan(built, n_trials=args.scan_trials, seed=args.seed + 5000)
    if "duration-scan" in checks:
        report["duration_scan"] = duration_scan(built, seed=args.seed + 6000)

    report["total_wall_time_s"] = time.monotonic() - t0

    text = json.dumps(report, indent=2, default=asdict)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out} ({len(text):,} bytes)")
    else:
        print(text)


if __name__ == "__main__":
    main()
