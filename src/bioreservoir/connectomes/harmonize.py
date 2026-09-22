"""Harmonized, comparable graphs for MaleCNS vs BANC (see docs/DATA.md "Harmonized graphs").

`malecns.py` and `banc.py` return each dataset's own native rules, and those rules are not
directly comparable:

- MaleCNS's `load_neurons` already keeps only `status == "Traced"` bodies, but its edge table
  (`connectome-weights-*.feather`) has **no synapse-count floor** (`weight >= 1`) and is not
  restricted to Traced-Traced pairs.
- BANC's `load_neurons` keeps every annotated row regardless of proofreading status (BANC has no
  single "Traced" flag baked into the annotation table the way MaleCNS does), while its edge
  table (`connections_princeton.csv.gz`) is a FlyWire-Codex curated export that already excludes
  pairs below 3 total synapses and all autapses (confirmed via the Dataverse file description,
  2026-09-18) — a threshold MaleCNS's edge table does not share.

`load_neurons` / `load_graph` apply ONE identical rule to both datasets instead:

    neurons = proofread neurons only
        MaleCNS: `status == "Traced"` (already what `malecns.load_neurons` returns).
        BANC:    `neuron_id` present in `backbone_proofread.tab` (CAVE materialization 626,
                 the dataset's own proofreading-status export — see `_banc_proofread_ids`).
    edges   = pairs where BOTH `pre_id` and `post_id` are in that neuron set, AND
              `syn_count >= min_syn` (default 5).

BANC's `connections_princeton.csv.gz` already drops pairs below 3 synapses server-side, so
`min_syn` must be >= `MIN_BANC_SYN_FLOOR` (3) for the two graphs to share one real threshold; a
lower value would only be a real threshold on the MaleCNS side.

`load_graph` also adds a `signed_weight` column (`sign(pre-synaptic neuron) * syn_count`, using
each neuron's `sign` from the common schema — see `schema.py`) so downstream code can build a
weighted graph without a second join.

Both functions cache their result as parquet under `data/processed/<dataset>-min<N>/`
(gitignored via `data/` in .gitignore) — re-running `load_graph` reads the cache instead of
re-streaming the 1 GB MaleCNS feather file or re-parsing BANC's CSVs.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyarrow import csv, ipc

from . import banc, malecns

REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

BACKBONE_PROOFREAD = banc.DATA_DIR / "backbone_proofread.tab"

DATASETS = ("malecns", "banc")

# `connections_princeton.csv.gz`'s own description (Harvard Dataverse file id 11842997, checked
# 2026-09-18): "Connections with fewer than 3 total synapses are excluded, along with autapses."
# A `min_syn` below this floor would not be a real threshold on the BANC side.
MIN_BANC_SYN_FLOOR = 3

DEFAULT_MIN_SYN = 5

HARMONIZED_EDGE_SCHEMA = pa.schema(
    [
        pa.field("pre_id", pa.int64()),
        pa.field("post_id", pa.int64()),
        pa.field("syn_count", pa.int64()),
        pa.field("signed_weight", pa.int64()),
    ]
)


def _banc_proofread_ids() -> pa.Array:
    """Deduplicated set of BANC neuron IDs marked proofread (materialization 626).

    `backbone_proofread.tab` lists one row per proofreading *annotation* — a neuron can be
    annotated more than once (e.g. separately for dendrite and axon backbone: 159,861 rows for
    115,151 unique `pt_root_id`s, checked 2026-09-18). Every row in this release has
    `proofread == "TRUE"`; the table only records proofread points, so presence is itself the
    proofread signal (there is no `FALSE` row to filter out, but the equality check is kept
    explicit in case a future materialization adds one).
    """
    table = csv.read_csv(
        BACKBONE_PROOFREAD,
        parse_options=csv.ParseOptions(delimiter="\t"),
        convert_options=csv.ConvertOptions(include_columns=["pt_root_id", "proofread"]),
    )
    # pyarrow's CSV type inference reads the quoted "TRUE"/"FALSE" literals as an actual bool
    # column, not a string column — compare against the Python bool, not the string "TRUE".
    table = table.filter(pc.equal(table["proofread"], True))
    return pc.unique(table["pt_root_id"])


def _dedupe_by_neuron_id(neurons: pa.Table) -> pa.Table:
    """Keep the first row per `neuron_id`.

    `codex_annotations_flat_table.tab` has 5 duplicate `pt_root_id`s out of 114,461 rows
    (checked 2026-09-18); left as-is this would fan a handful of neurons into two rows and
    double-count them in population and edge joins below.
    """
    # pyarrow has no direct "first index per unique key" op; do the small (<= a few hundred
    # thousand rows) case with a Python-level pass instead of pulling in pandas here.
    seen: set[int] = set()
    keep_mask = []
    for neuron_id in neurons.column("neuron_id").to_pylist():
        if neuron_id in seen:
            keep_mask.append(False)
        else:
            seen.add(neuron_id)
            keep_mask.append(True)
    return neurons.filter(pa.array(keep_mask, type=pa.bool_()))


def load_neurons(dataset: str) -> pa.Table:
    """Proofread-only neurons for `dataset`, in the common schema (`schema.py`).

    MaleCNS: `malecns.load_neurons()` already filters to `status == "Traced"`.
    BANC: `banc.load_neurons()` filtered down to IDs present in `_banc_proofread_ids()`.
    """
    if dataset == "malecns":
        return _dedupe_by_neuron_id(malecns.load_neurons())
    if dataset == "banc":
        neurons = banc.load_neurons()
        proofread_ids = _banc_proofread_ids()
        neurons = neurons.filter(pc.is_in(neurons["neuron_id"], value_set=proofread_ids))
        return _dedupe_by_neuron_id(neurons)
    raise ValueError(f"unknown dataset: {dataset!r} (expected one of {DATASETS})")


def _filter_edge_table(edges: pa.Table, neuron_ids: pa.Array, min_syn: int) -> pa.Table:
    mask = pc.and_(
        pc.and_(
            pc.is_in(edges["pre_id"], value_set=neuron_ids),
            pc.is_in(edges["post_id"], value_set=neuron_ids),
        ),
        pc.greater_equal(edges["syn_count"], min_syn),
    )
    return edges.filter(mask)


def _stream_malecns_edges(neuron_ids: pa.Array, min_syn: int) -> pa.Table:
    """Filter MaleCNS's 151.9M-row edge table one IPC record batch at a time.

    The raw feather file is Arrow IPC with 2,318 record batches of 65,536 rows each (checked
    2026-09-18); reading it via `malecns.load_edges()` materializes all 151.9M rows in memory at
    once (~3.6 GB just for the three int64 columns, more during the filter step). Streaming
    batch-by-batch keeps peak memory bounded by one batch (~1.5 MB) plus the already-filtered
    rows collected so far, which are always far fewer than 151.9M once both proofread-only and
    `min_syn` are applied.
    """
    pieces: list[pa.Table] = []
    with pa.memory_map(str(malecns.CONNECTOME_WEIGHTS), "rb") as source:
        reader = ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            table = pa.Table.from_batches([batch]).rename_columns(
                ["pre_id", "post_id", "syn_count"]
            )
            filtered = _filter_edge_table(table, neuron_ids, min_syn)
            if filtered.num_rows:
                pieces.append(filtered)
    if not pieces:
        return pa.table(
            {
                "pre_id": pa.array([], type=pa.int64()),
                "post_id": pa.array([], type=pa.int64()),
                "syn_count": pa.array([], type=pa.int64()),
            }
        )
    return pa.concat_tables(pieces)


def _load_filtered_edges(dataset: str, neuron_ids: pa.Array, min_syn: int) -> pa.Table:
    if dataset == "malecns":
        return _stream_malecns_edges(neuron_ids, min_syn)
    # BANC's edge table is already small (2.68M rows) — no streaming needed.
    return _filter_edge_table(banc.load_edges(), neuron_ids, min_syn)


def load_graph(dataset: str, min_syn: int = DEFAULT_MIN_SYN) -> tuple[pa.Table, pa.Table]:
    """Harmonized (neurons, edges) for `dataset` ("malecns" or "banc"). See module docstring."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset!r} (expected one of {DATASETS})")
    if min_syn < MIN_BANC_SYN_FLOOR:
        raise ValueError(
            f"min_syn={min_syn} is below BANC's own connections_princeton.csv.gz floor "
            f"({MIN_BANC_SYN_FLOOR} synapses/pair, server-side) — the two graphs would no "
            f"longer share one identical rule; use min_syn >= {MIN_BANC_SYN_FLOOR}."
        )

    cache_dir = PROCESSED_DIR / f"{dataset}-min{min_syn}"
    neurons_path = cache_dir / "neurons.parquet"
    edges_path = cache_dir / "edges.parquet"
    if neurons_path.exists() and edges_path.exists():
        return pq.read_table(neurons_path), pq.read_table(edges_path)

    neurons = load_neurons(dataset)
    neuron_ids = neurons["neuron_id"]
    sign_lookup = neurons.select(["neuron_id", "sign"]).rename_columns(["pre_id", "pre_sign"])

    edges = _load_filtered_edges(dataset, neuron_ids, min_syn)
    edges = edges.join(sign_lookup, keys="pre_id", join_type="left outer")
    signed_weight = pc.multiply(pc.cast(edges["pre_sign"], pa.int64()), edges["syn_count"])
    edges = edges.drop_columns(["pre_sign"]).append_column("signed_weight", signed_weight)
    edges = edges.select(HARMONIZED_EDGE_SCHEMA.names).cast(HARMONIZED_EDGE_SCHEMA)

    cache_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(neurons, neurons_path)
    pq.write_table(edges, edges_path)
    return neurons, edges
