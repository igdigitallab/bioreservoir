"""Loader for Janelia MaleCNS v1.0 (adult male Drosophila CNS, CC BY 4.0) into the common
connectome schema (see `schema.py`).

Source files (fetched by `scripts/fetch_data.py` into `data/raw/malecns-v1.0/`):
    body-annotations-male-cns-v1.0-minconf-0.5.feather
        One row per segmented body: cell type/class hierarchy, soma side, proofreading
        `status`. We keep only `status == "Traced"` (fully proofread neurons; ~165k of the
        211k rows), matching the headline "neuron" count reported for this release.
    body-neurotransmitters-male-cns-v1.0.feather
        One row per body with synapses (finer-grained than body-annotations: 1.8M rows,
        includes untyped/fragment bodies): per-body predicted neurotransmitter and its
        confidence.
    connectome-weights-male-cns-v1.0-minconf-0.5.feather
        The full segment-to-segment connection graph (`body_pre`, `body_post`, `weight`),
        *not* restricted to Traced/typed bodies — see `load_edges` docstring.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import feather

from .schema import EDGE_SCHEMA, NEURON_SCHEMA, nt_to_sign

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "raw" / "malecns-v1.0"

BODY_ANNOTATIONS = DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
BODY_NEUROTRANSMITTERS = DATA_DIR / "body-neurotransmitters-male-cns-v1.0.feather"
CONNECTOME_WEIGHTS = DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

SEX = "male"


def load_neurons() -> pa.Table:
    """Traced (fully proofread) neurons, joined with their predicted neurotransmitter.

    Rows with `status != "Traced"` (orphan fragments, glia, unimportant/anchor bookkeeping
    bodies) are dropped, since they are not analysed neurons.
    """
    # NOTE: feather.read_table(columns=...) returns columns in the *file's* native order, not
    # the order requested — an explicit .select() afterwards is required to pin the order
    # before a positional rename_columns(), otherwise fields get silently swapped.
    annotations = feather.read_table(
        BODY_ANNOTATIONS,
        columns=["bodyId", "type", "class", "superclass", "somaSide", "status"],
    )
    annotations = annotations.filter(pc.equal(annotations["status"], "Traced"))
    annotations = annotations.select(["bodyId", "type", "class", "superclass", "somaSide"])
    annotations = annotations.rename_columns(
        ["neuron_id", "cell_type", "cell_class", "super_class", "side"]
    )

    nt = feather.read_table(
        BODY_NEUROTRANSMITTERS,
        columns=["body", "predicted_nt", "predicted_nt_confidence"],
    )
    nt = nt.select(["body", "predicted_nt", "predicted_nt_confidence"])
    nt = nt.rename_columns(["neuron_id", "nt", "nt_conf"])

    neurons = annotations.join(nt, keys="neuron_id", join_type="left outer")

    sex_col = pa.array([SEX] * neurons.num_rows, type=pa.string())
    sign_col = pa.array(
        [nt_to_sign(x) for x in neurons.column("nt").to_pylist()], type=pa.int8()
    )
    neurons = neurons.append_column("sex", sex_col).append_column("sign", sign_col)

    return neurons.select(NEURON_SCHEMA.names).cast(NEURON_SCHEMA)


def load_edges() -> pa.Table:
    """The full body-to-body connection graph, with synapse counts.

    This is *not* restricted to the Traced neurons returned by `load_neurons` — it is the
    complete segment-to-segment graph as published (includes orphan fragments and glia, which
    can be legitimate pre/post partners of a traced neuron). Callers that need only the
    subgraph between typed neurons should inner-join both `pre_id` and `post_id` against
    `load_neurons()["neuron_id"]`.
    """
    weights = feather.read_table(CONNECTOME_WEIGHTS, columns=["body_pre", "body_post", "weight"])
    weights = weights.rename_columns(["pre_id", "post_id", "syn_count"])
    return weights.cast(EDGE_SCHEMA)
