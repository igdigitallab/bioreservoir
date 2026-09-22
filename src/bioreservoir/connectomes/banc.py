"""Loader for BANC (Brain And Nerve Cord, adult female Drosophila, CC BY 4.0) into the common
connectome schema (see `schema.py`).

Source files (fetched by `scripts/fetch_data.py` into `data/raw/banc-626/`, all keyed on CAVE
materialization **626**, the stable snapshot this Dataverse release is built on):
    codex_annotations_flat_table.tab
        One row per neuron carrying a FlyWire-Codex annotation: cell type/class hierarchy and
        soma side. Only a subset of the ~188k segmented bodies have a `cell_type` at all (see
        `load_neurons` docstring) — this is *not* the full body count.
    banc_neurotransmitter_prediction.csv
        One row per neuron with a synapse-based neurotransmitter prediction: predicted label
        and a 0-100 confidence score, from the same prediction pipeline (one source table for
        both fields, matching how MaleCNS's `body-neurotransmitters` table pairs them).
    connections_princeton.csv.gz
        Synapse counts per (pre, post, neuropil) triple; summed over neuropil to produce one
        row per (pre, post) pair, matching MaleCNS's already-aggregated edge table.

Caution: quoted "NA" values are a real null, not the two-letter string "NA" — always parse with
`strings_can_be_null=True, null_values=["NA", ""]` (as this module does), otherwise every row
looks non-null and coverage figures are wrong.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import csv

from .schema import EDGE_SCHEMA, NEURON_SCHEMA, nt_to_sign

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "raw" / "banc-626"

CODEX_ANNOTATIONS = DATA_DIR / "codex_annotations_flat_table.tab"
NEUROTRANSMITTER_PREDICTION = DATA_DIR / "banc_neurotransmitter_prediction.csv"
CONNECTIONS = DATA_DIR / "connections_princeton.csv.gz"

SEX = "female"


def load_neurons() -> pa.Table:
    """All annotated BANC neurons, joined with their predicted neurotransmitter.

    `codex_annotations_flat_table.tab` covers every neuron that received a FlyWire-Codex
    annotation pass, but only a fraction of those rows have a non-null `cell_type` (the rest
    are annotated at a coarser level, e.g. only `region`/`side`, or not yet typed) — unlike
    MaleCNS, we do not filter these out here, since BANC has no single "Traced" proofreading
    flag in this table; `cell_type` nullness itself is the coverage signal (see docs/DATA.md).
    """
    annotations = csv.read_csv(
        CODEX_ANNOTATIONS,
        parse_options=csv.ParseOptions(delimiter="\t"),
        convert_options=csv.ConvertOptions(
            include_columns=["pt_root_id", "cell_type", "cell_class", "super_class", "side"],
            strings_can_be_null=True,
            null_values=["NA", ""],
        ),
    )
    annotations = annotations.rename_columns(
        ["neuron_id", "cell_type", "cell_class", "super_class", "side"]
    )
    # BANC spells sides out ("left"/"right"/"midline"); the common schema uses MaleCNS's "L"/"R"/"M".
    side = annotations["side"]
    for raw, short in (("left", "L"), ("right", "R"), ("midline", "M")):
        side = pc.replace_substring_regex(side, pattern=f"^{raw}$", replacement=short)
    annotations = annotations.set_column(annotations.schema.get_field_index("side"), "side", side)

    nt = csv.read_csv(
        NEUROTRANSMITTER_PREDICTION,
        convert_options=csv.ConvertOptions(
            include_columns=["root_626", "neurotransmitter_predicted", "neurotransmitter_score"],
            strings_can_be_null=True,
            null_values=["NA", ""],
        ),
    )
    nt = nt.rename_columns(["neuron_id", "nt", "nt_conf_pct"])
    # A handful of neuron_ids (13 of 143,122) appear twice in the source file; keep the first
    # occurrence so the join below cannot fan a neuron out into two rows.
    nt = nt.to_pandas().drop_duplicates(subset="neuron_id", keep="first")
    nt = pa.Table.from_pandas(nt, preserve_index=False)
    # Rescale 0-100 confidence to the [0, 1] scale used by MaleCNS's predicted_nt_confidence.
    nt = nt.append_column(
        "nt_conf", pc.divide(pc.cast(nt["nt_conf_pct"], pa.float64()), 100.0)
    )
    nt = nt.drop_columns(["nt_conf_pct"])

    neurons = annotations.join(nt, keys="neuron_id", join_type="left outer")

    sex_col = pa.array([SEX] * neurons.num_rows, type=pa.string())
    sign_col = pa.array(
        [nt_to_sign(x) for x in neurons.column("nt").to_pylist()], type=pa.int8()
    )
    neurons = neurons.append_column("sex", sex_col).append_column("sign", sign_col)

    return neurons.select(NEURON_SCHEMA.names).cast(NEURON_SCHEMA)


def load_edges() -> pa.Table:
    """Body-to-body synapse counts, summed across neuropils.

    The source file lists one row per (pre, post, neuropil) triple (a connection that spans
    several neuropils is split across several rows); we sum `syn_count` over neuropil to get
    one row per (pre, post) pair, matching MaleCNS's already-aggregated edge table. Neuropil
    identity is discarded here — callers who need it should read `connections_princeton.csv.gz`
    directly.
    """
    raw = csv.read_csv(
        CONNECTIONS,
        convert_options=csv.ConvertOptions(
            include_columns=["pre_root_id", "post_root_id", "syn_count"]
        ),
    )
    grouped = raw.group_by(["pre_root_id", "post_root_id"]).aggregate([("syn_count", "sum")])
    grouped = grouped.rename_columns(["pre_id", "post_id", "syn_count"])
    return grouped.cast(EDGE_SCHEMA)
