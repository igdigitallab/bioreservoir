"""Read-only summary of a loaded connectome: counts, NT distribution, key cell types.

Usage:
    python -m bioreservoir.connectomes.inspect malecns
    python -m bioreservoir.connectomes.inspect banc
    python -m bioreservoir.connectomes.inspect malecns --harmonized [--min-syn 5]

`--harmonized` prints the harmonized, cross-brain-comparable summary instead (proofread-only
neurons, one shared min-synapse edge threshold, signed weights, population counts) — see
`harmonize.py` and `populations.py` for the rules applied.

Only reads data (loaders + pyarrow aggregation); never builds or runs a simulation.
"""

from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.compute as pc

from . import banc, harmonize, malecns, populations

# (label, cell_type values to match, match mode). "exact" matches `cell_type` against the set
# verbatim; "startswith" matches any `cell_type` beginning with the given prefix. Patterns were
# chosen by inspecting the actual `cell_type`/`type` vocabulary of each dataset (see
# docs/DATA.md) rather than assumed — several targets below have no dataset-native name and are
# reported as absent (0) rather than matched to a lookalike.
CELL_TYPE_QUERIES: list[tuple[str, set[str], str]] = [
    ("sugar GRN (exact gene name Gr5a/Gr64f)", {"Gr5a", "Gr64f"}, "exact"),
    ("bitter GRN (exact gene name Gr66a)", {"Gr66a"}, "exact"),
    ("MN9 (proboscis motor neuron)", {"MN9"}, "exact"),
    ("giant fiber DNp01", {"DNp01"}, "exact"),
    ("DNa01", {"DNa01"}, "exact"),
    ("DNa02", {"DNa02"}, "exact"),
    ("MDN (moonwalker descending neuron)", {"MDN"}, "exact"),
    ("P9 descending neuron", {"P9"}, "exact"),
    ("P1 / pC1 courtship cluster (via pC1 prefix, see docs/DATA.md)", {"pC1"}, "startswith"),
    ("pIP10", {"pIP10"}, "exact"),
]

# Class-level proxy for taste GRNs: neither dataset names gustatory sensory neurons after their
# receptor gene (see CELL_TYPE_QUERIES above), so as extra context we also count the broader
# class each dataset uses for taste neurons. Not a substitute for the exact-gene-name counts.
GUSTATORY_CLASS_PROXY = {
    "malecns": ("cell_class", "gustatory"),
    "banc": ("cell_class", "taste_peg_neuron"),
}

LOADERS = {"malecns": malecns, "banc": banc}


def count_cell_type(neurons: pa.Table, values: set[str], mode: str) -> int:
    col = neurons.column("cell_type")
    if mode == "exact":
        mask = pc.is_in(col, value_set=pa.array(sorted(values)))
    elif mode == "startswith":
        (prefix,) = values
        mask = pc.starts_with(pc.fill_null(col, ""), prefix)
    else:
        raise ValueError(f"unknown match mode: {mode}")
    return pc.sum(mask.cast(pa.int64())).as_py() or 0


def run(dataset: str) -> None:
    module = LOADERS[dataset]
    neurons = module.load_neurons()
    edges = module.load_edges()

    print(f"=== {dataset} ===")
    print(f"neurons: {neurons.num_rows:,}")
    print(f"edges:   {edges.num_rows:,}")
    total_synapses = pc.sum(edges.column("syn_count")).as_py()
    print(f"total synapses (sum of syn_count over all edges): {total_synapses:,}")

    print("\nneurotransmitter distribution (predicted, neuron-level):")
    nt_counts = neurons.group_by(["nt"]).aggregate([("neuron_id", "count")])
    nt_counts = nt_counts.sort_by([("neuron_id_count", "descending")])
    for row in nt_counts.to_pylist():
        label = row["nt"] if row["nt"] is not None else "(null)"
        print(f"  {label:<16} {row['neuron_id_count']:>8,}")

    print("\nkey cell types:")
    for label, values, mode in CELL_TYPE_QUERIES:
        n = count_cell_type(neurons, values, mode)
        status = "present" if n > 0 else "NOT FOUND"
        print(f"  {label:<55} {n:>6,}  [{status}]")

    class_col, class_value = GUSTATORY_CLASS_PROXY[dataset]
    class_count = pc.sum(
        pc.equal(neurons.column(class_col), class_value).cast(pa.int64())
    ).as_py()
    print(
        f"\n  (context, not a receptor-specific match) {class_col}=='{class_value}': "
        f"{class_count:,} neurons — candidate taste GRNs, see docs/DATA.md"
    )


def run_harmonized(dataset: str, min_syn: int) -> None:
    """Harmonized summary: proofread-only neurons, one shared min-synapse edge threshold,
    signed weights and population counts. See `harmonize.py` / `populations.py`.
    """
    neurons, edges = harmonize.load_graph(dataset, min_syn=min_syn)

    print(f"=== {dataset} (harmonized, min_syn={min_syn}) ===")
    print(f"neurons: {neurons.num_rows:,}")
    print(f"edges:   {edges.num_rows:,}")
    total_synapses = pc.sum(edges.column("syn_count")).as_py() or 0
    print(f"total synapses (sum of syn_count over harmonized edges): {total_synapses:,}")

    # Sum of out-degree over all neurons == sum of in-degree over all neurons == edge count, so
    # mean in-degree and mean out-degree are the same number by construction; reported once.
    mean_degree = edges.num_rows / neurons.num_rows if neurons.num_rows else 0.0
    print(f"mean degree (edges / neurons, in == out by construction): {mean_degree:.2f}")

    signed_weight = edges.column("signed_weight")
    excitatory = pc.sum(pc.greater(signed_weight, 0).cast(pa.int64())).as_py() or 0
    inhibitory = pc.sum(pc.less(signed_weight, 0).cast(pa.int64())).as_py() or 0
    unknown = pc.sum(pc.equal(signed_weight, 0).cast(pa.int64())).as_py() or 0
    print("\nE/I sign split (by presynaptic neuron's predicted neurotransmitter):")
    print(f"  excitatory edges (signed_weight > 0): {excitatory:>10,}")
    print(f"  inhibitory edges (signed_weight < 0): {inhibitory:>10,}")
    print(f"  unknown/modulatory (signed_weight = 0): {unknown:>8,}")

    print("\npopulations (see experiments/001-fly-oracle/populations.yaml):")
    for row in populations.report(dataset):
        print(f"  {row.name:<32} role={row.role:<8} n={row.count:>6}  {row.rule_summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(LOADERS))
    parser.add_argument(
        "--harmonized",
        action="store_true",
        help="print the harmonized, cross-brain-comparable summary instead of the raw one",
    )
    parser.add_argument(
        "--min-syn",
        type=int,
        default=harmonize.DEFAULT_MIN_SYN,
        help=f"minimum synapses/pair for --harmonized (default {harmonize.DEFAULT_MIN_SYN})",
    )
    args = parser.parse_args()
    if args.harmonized:
        run_harmonized(args.dataset, args.min_syn)
    else:
        run(args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
