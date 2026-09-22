"""Benchmark + sanity checks for `sim.lif.LIFNetwork` on the real connectomes (and a synthetic
fallback). Entry point:

    python -m bioreservoir.sim.bench <malecns|banc> [--synthetic] [--min-syn N]
        [--n-trials N] [--trial-ms N] [--codegen-target cython|numpy]

Must be run inside the memory/CPU cage described in experiments/001-fly-oracle/RUNNING.md — see `docs/MODEL.md` for the exact
command used to produce the numbers there. This module does not build the cage
itself (a plain Python process has no way to sandbox its own memory), it only assumes it is
already running inside one.
"""

from __future__ import annotations

import argparse
import resource
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from bioreservoir.connectomes import harmonize, populations
from bioreservoir.sim.lif import DEFAULT_POISSON_RATE_HZ, LIFNetwork

SYNTHETIC_N_NEURONS = 165_000
SYNTHETIC_N_EDGES = 20_000_000
SYNTHETIC_EXCITATORY_FRACTION = 0.8

# Raw per-dataset column + value used to split a population by hemisphere (see
# populations.py / populations.yaml header: each dataset keeps its own vocabulary). MaleCNS's
# `somaSide` already has sensory neurons' nulls filled from `rootSide` inside
# `populations.raw_annotations` (see `_malecns_side_with_root_fallback` there), so both datasets
# use a single raw column here — no per-dataset lookup needed.
SIDE_COLUMN = {"malecns": "somaSide", "banc": "side"}
SIDE_VALUE = {"malecns": {"left": "L", "right": "R"}, "banc": {"left": "left", "right": "right"}}

# Populations used by the sanity checks below (see experiments/001-fly-oracle/populations.yaml).
# Both datasets now resolve gustatory_sugar via the cross-dataset LB3a/LB3b type match (see
# populations.yaml), so the Shiu-style sugar-GRN -> MN9 check runs on the same ~20-30-neuron
# scale Shiu used, instead of MaleCNS's much larger, sugar/bitter-unsplit gustatory_all (n=1,428).
GUSTATORY_INPUT = {"malecns": "gustatory_sugar", "banc": "gustatory_sugar"}
LATERAL_INPUT_POPULATION = "mechanosensory_johnstons_organ"
TRIAL_MS_DEFAULT = 1000.0


def peak_rss_mb() -> float:
    """High-water-mark resident set size for this process so far, in MB.

    `ru_maxrss` is tracked by the kernel for the process's whole lifetime (Linux: KB), so this
    reflects the true peak regardless of when it is called — no polling loop needed.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def synthetic_graph(
    n_neurons: int = SYNTHETIC_N_NEURONS,
    n_edges: int = SYNTHETIC_N_EDGES,
    excitatory_fraction: float = SYNTHETIC_EXCITATORY_FRACTION,
    seed: int = 0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """A MaleCNS-scale random directed graph: `n_neurons` nodes, `n_edges` edges, signed weight
    from an 80/20 excitatory/inhibitory split assigned per presynaptic neuron (not per edge, to
    match the real data's convention that sign is a neuron property — see `schema.py`)."""
    rng = np.random.default_rng(seed)
    pre_idx = rng.integers(0, n_neurons, size=n_edges, dtype=np.int64)
    post_idx = rng.integers(0, n_neurons, size=n_edges, dtype=np.int64)
    neuron_sign = np.where(rng.random(n_neurons) < excitatory_fraction, 1.0, -1.0)
    syn_count = rng.integers(5, 30, size=n_edges).astype(np.float64)  # >= harmonize's min_syn=5
    weight = neuron_sign[pre_idx] * syn_count
    return n_neurons, pre_idx, post_idx, weight


def graph_to_arrays(
    dataset: str, min_syn: int = 5
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Real, harmonized graph as dense arrays for `LIFNetwork`, plus a neuron_id -> dense-index
    map (needed to translate named populations into Brian2 indices, see `population_indices`)."""
    neurons, edges = harmonize.load_graph(dataset, min_syn=min_syn)
    dense_idx = pa.array(np.arange(neurons.num_rows, dtype=np.int64))
    index_lookup = pa.table({"neuron_id": neurons.column("neuron_id"), "dense_idx": dense_idx})

    edges = edges.join(
        index_lookup.rename_columns(["pre_id", "pre_idx"]), keys="pre_id", join_type="inner"
    )
    edges = edges.join(
        index_lookup.rename_columns(["post_id", "post_idx"]), keys="post_id", join_type="inner"
    )

    pre_idx = edges.column("pre_idx").to_numpy()
    post_idx = edges.column("post_idx").to_numpy()
    weight = edges.column("signed_weight").to_numpy().astype(np.float64)
    id_to_dense = dict(zip(neurons.column("neuron_id").to_pylist(), range(neurons.num_rows)))
    return neurons.num_rows, pre_idx, post_idx, weight, id_to_dense


def population_indices(
    dataset: str, id_to_dense: dict[int, int], population_name: str, side: str | None = None
) -> np.ndarray | None:
    """Dense Brian2 indices for a named population (optionally restricted to one hemisphere).

    Returns `None` if the population is `absent` for `dataset` (see populations.yaml). `side` is
    `"left"` or `"right"`, translated to each dataset's own raw value via `SIDE_VALUE` and
    checked against `SIDE_COLUMN[dataset]` on the table `populations.raw_annotations` already
    exposes (MaleCNS's `somaSide` there already has sensory neurons' nulls filled from
    `rootSide`, see `populations._malecns_side_with_root_fallback`).
    """
    rule = populations.load_rules()[population_name].get(dataset)
    if rule is None or rule.absent:
        return None
    table = populations.raw_annotations(dataset)
    mask = rule.mask(table)
    if side is not None:
        raw_value = SIDE_VALUE[dataset][side]
        side_mask = pc.equal(table.column(SIDE_COLUMN[dataset]), raw_value)
        mask = pc.and_(mask, side_mask)
    ids = table.filter(mask).column("neuron_id").to_pylist()
    return np.array([id_to_dense[i] for i in ids if i in id_to_dense], dtype=np.int64)


def balanced_lateral_population(
    dataset: str, id_to_dense: dict[int, int], population_name: str, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Equal-size left/right index arrays for `population_name`.

    BANC's proofreading is 45% denser on the right side than the left (66,745 vs 46,107 neurons
    overall), which would make a raw left-vs-right stimulation comparison confound "which side
    got proofread more" with "which side the model turns toward". Whichever side has more
    candidate neurons is downsampled (fixed seed, without replacement) to match the smaller
    side, and the counts actually used are returned for the report.
    """
    left = population_indices(dataset, id_to_dense, population_name, side="left")
    right = population_indices(dataset, id_to_dense, population_name, side="right")
    n = min(len(left), len(right))
    rng = np.random.default_rng(seed)
    left_used = rng.choice(left, size=n, replace=False) if len(left) > n else left
    right_used = rng.choice(right, size=n, replace=False) if len(right) > n else right
    counts = {"left_available": len(left), "right_available": len(right), "used_per_side": n}
    return left_used, right_used, counts


def run_benchmark(
    net: LIFNetwork, n_trials: int, trial_ms: float, seed: int = 0
) -> dict[str, float]:
    """Warm up (triggers codegen compilation, excluded from timing), then time `n_trials`
    empty-input trials to measure steady-state wall-clock per simulated second."""
    net.run_trial({}, duration_ms=10.0, seed=seed)  # warm-up: compiles, not timed
    wall_times = []
    for i in range(n_trials):
        result = net.run_trial({}, duration_ms=trial_ms, seed=seed + 1 + i)
        wall_times.append(result.wall_time_s)
    mean_wall_s = float(np.mean(wall_times))
    return {
        "n_trials": n_trials,
        "trial_ms": trial_ms,
        "mean_wall_s_per_trial": mean_wall_s,
        "sec_wall_per_sec_simulated": mean_wall_s / (trial_ms / 1000.0),
    }


def spike_rate_hz(count: int, duration_ms: float) -> float:
    return count / (duration_ms / 1000.0)


def run_sanity_checks(
    dataset: str, net: LIFNetwork, id_to_dense: dict[int, int], trial_ms: float, seed: int = 100
) -> dict:
    results: dict = {}

    # (c) no-input baseline: spontaneous activity should be ~0 in a connectome-only LIF model
    # (no external drive, no gap junctions, no intrinsic pacemaker current in the equations).
    baseline = net.run_trial({}, duration_ms=trial_ms, seed=seed)
    results["baseline_total_spikes"] = int(baseline.spike_counts.sum())
    results["baseline_mean_rate_hz"] = spike_rate_hz(
        baseline.spike_counts.sum() / net.n_neurons, trial_ms
    )

    # (a) Shiu-style: drive gustatory input, look for MN9 (proboscis extension) above baseline.
    # BANC has no confirmed MN9 homolog (populations.yaml: MN9_proboscis is `absent` there), so
    # this check only produces a result for MaleCNS.
    mn9_idx = population_indices(dataset, id_to_dense, "MN9_proboscis")
    gustatory_idx = population_indices(dataset, id_to_dense, GUSTATORY_INPUT[dataset])
    if mn9_idx is not None and len(mn9_idx) and gustatory_idx is not None and len(gustatory_idx):
        driven = net.run_trial(
            (gustatory_idx, np.full(len(gustatory_idx), DEFAULT_POISSON_RATE_HZ)),
            duration_ms=trial_ms,
            seed=seed + 1,
        )
        results["gustatory_drive"] = {
            "input_population": GUSTATORY_INPUT[dataset],
            "n_driven": len(gustatory_idx),
            "rate_hz": DEFAULT_POISSON_RATE_HZ,
            "mn9_spikes": driven.spike_counts[mn9_idx].tolist(),
            "mn9_baseline_spikes": baseline.spike_counts[mn9_idx].tolist(),
            "mn9_above_baseline": bool((driven.spike_counts[mn9_idx] > baseline.spike_counts[mn9_idx]).any()),
        }
    else:
        results["gustatory_drive"] = None  # MN9 or gustatory input absent for this dataset

    # (b) T-maze asymmetry: drive left- vs right-side olfactory receptor neurons (equal counts
    # per side, see `balanced_lateral_population`), compare DNa01/DNa02 left/right firing.
    left_idx, right_idx, side_counts = balanced_lateral_population(
        dataset, id_to_dense, LATERAL_INPUT_POPULATION, seed=seed
    )
    readout_idx = {
        name: population_indices(dataset, id_to_dense, name)
        for name in ("DNa01_left", "DNa01_right", "DNa02_left", "DNa02_right")
    }
    tmaze: dict = {"input_population": LATERAL_INPUT_POPULATION, "side_counts": side_counts}
    for side_name, idx in (("left_driven", left_idx), ("right_driven", right_idx)):
        if idx.size == 0:
            tmaze[side_name] = None
            continue
        trial = net.run_trial(
            (idx, np.full(len(idx), DEFAULT_POISSON_RATE_HZ)),
            duration_ms=trial_ms,
            seed=seed + 2 if side_name == "left_driven" else seed + 3,
        )
        tmaze[side_name] = {
            readout: int(trial.spike_counts[readout_idx[readout]].sum())
            for readout in readout_idx
            if readout_idx[readout] is not None
        }
    results["t_maze"] = tmaze

    return results


def _print_report(dataset: str, build: dict, bench: dict, sanity: dict) -> None:
    print(f"\n=== {dataset} ===")
    print(f"neurons={build['n_neurons']:,}  synapses={build['n_synapses']:,}")
    print(f"codegen target: {build['codegen_target']}")
    print(f"build time: {build['build_time_s']:.1f} s")
    print(f"peak RSS so far: {peak_rss_mb():.0f} MB")
    print(
        f"benchmark: {bench['n_trials']} trials x {bench['trial_ms']:.0f} ms, "
        f"mean wall time/trial = {bench['mean_wall_s_per_trial']:.3f} s "
        f"({bench['sec_wall_per_sec_simulated']:.3f} s wall / s simulated)"
    )
    print(
        f"\nno-input baseline: {sanity['baseline_total_spikes']} total spikes over "
        f"{build['n_neurons']:,} neurons ({sanity['baseline_mean_rate_hz']:.4f} Hz mean)"
    )
    gustatory = sanity["gustatory_drive"]
    if gustatory is None:
        print("gustatory -> MN9: skipped (MN9 or gustatory input absent for this dataset)")
    else:
        print(
            f"gustatory ({gustatory['input_population']}, n={gustatory['n_driven']}, "
            f"{gustatory['rate_hz']:.0f} Hz) -> MN9 spikes: {gustatory['mn9_spikes']} "
            f"(baseline {gustatory['mn9_baseline_spikes']}) "
            f"above baseline: {gustatory['mn9_above_baseline']}"
        )
    tmaze = sanity["t_maze"]
    print(f"\nT-maze ({tmaze['input_population']}, per-side counts: {tmaze['side_counts']}):")
    print(f"  left-driven readout spikes:  {tmaze['left_driven']}")
    print(f"  right-driven readout spikes: {tmaze['right_driven']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["malecns", "banc"])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--trial-ms", type=float, default=TRIAL_MS_DEFAULT)
    parser.add_argument("--codegen-target", choices=["cython", "numpy"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.monotonic()
    if args.synthetic:
        n_neurons, pre_idx, post_idx, weight = synthetic_graph(seed=args.seed)
        id_to_dense = {}
    else:
        n_neurons, pre_idx, post_idx, weight, id_to_dense = graph_to_arrays(
            args.dataset, min_syn=args.min_syn
        )
    load_time_s = time.monotonic() - t0

    net = LIFNetwork(
        n_neurons, pre_idx, post_idx, weight, codegen_target=args.codegen_target
    )
    build = {
        "n_neurons": net.n_neurons,
        "n_synapses": net.n_synapses,
        "codegen_target": net.codegen_target,
        "build_time_s": net.build_time_s + load_time_s,
    }

    bench = run_benchmark(net, n_trials=args.n_trials, trial_ms=args.trial_ms, seed=args.seed)

    if args.synthetic:
        print(f"\n=== {args.dataset} (SYNTHETIC graph, no populations) ===")
        print(f"neurons={build['n_neurons']:,}  synapses={build['n_synapses']:,}")
        print(f"codegen target: {build['codegen_target']}")
        print(f"build time: {build['build_time_s']:.1f} s")
        print(f"peak RSS so far: {peak_rss_mb():.0f} MB")
        print(
            f"benchmark: {bench['n_trials']} trials x {bench['trial_ms']:.0f} ms, "
            f"mean wall time/trial = {bench['mean_wall_s_per_trial']:.3f} s "
            f"({bench['sec_wall_per_sec_simulated']:.3f} s wall / s simulated)"
        )
        return

    sanity = run_sanity_checks(args.dataset, net, id_to_dense, trial_ms=1000.0, seed=args.seed + 100)
    _print_report(args.dataset, build, bench, sanity)


if __name__ == "__main__":
    main()
