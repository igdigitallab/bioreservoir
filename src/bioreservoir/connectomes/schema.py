"""Common schema shared by every connectome loader (MaleCNS, BANC, ...).

Each loader (`malecns.py`, `banc.py`) reads its dataset's native files and returns two
`pyarrow.Table` objects in this shape, so downstream code (encoder, simulation, readout)
never needs to know which brain it is looking at.

Neurons table columns:
    neuron_id  int64   dataset-native body/root ID
    cell_type  string  finest-grained annotation (e.g. "DNp01"); may be null if unassigned
    cell_class string  mid-level functional class; sparsely populated in both datasets
    super_class string broad category (sensory / intrinsic / descending / motor / ...)
    side       string  "L" / "R" / "M" (midline) / null
    nt         string  predicted fast neurotransmitter (gaba/glutamate/acetylcholine/dopamine/
                        serotonin/octopamine/histamine/tyramine/unclear), or null
    nt_conf    float64 confidence of `nt`, in [0, 1]; null if unavailable
    sex        string  "male" or "female" (constant per dataset — both source brains are
                        single-sex reconstructions)
    sign       int8    +1 excitatory / -1 inhibitory, derived from `nt` via NT_SIGN — Shiu et
                        al.'s binary rule, no "unknown" value (see caveats below)

Edges table columns:
    pre_id     int64   presynaptic neuron_id
    post_id    int64   postsynaptic neuron_id
    syn_count  int64   number of synapses supporting this connection

Sign mapping and its caveats
-----------------------------
`sign` follows Shiu et al.'s own rule verbatim (*Nature* 634, 2024, Methods, "Neurotransmitter
predictions", PMC11446845 — read as public paper text, never the paper's CC BY-NC FlyWire data):
"We assume GABAergic and glutamatergic neurons are inhibitory... and that each neuron is either
exclusively inhibitory or excitatory... Neurons predicted to be dopaminergic, octopaminergic or
serotonergic are assigned to the excitatory category." This is a **strict binary partition** —
Shiu's model has no third "unknown/modulatory" bucket; every neuron is either +1 or -1:

    gaba, glutamate      -> -1 (inhibitory; glutamate is inhibitory at the fly NMJ and in most
                                 central synapses via glutamate-gated chloride channels (GluCl) —
                                 the opposite of the vertebrate convention)
    everything else      -> +1 (excitatory by Shiu's rule: acetylcholine, dopamine, serotonin,
                                 octopamine, plus tyramine/histamine/unclear/null, which do not
                                 appear in Shiu's own classifier's output — see caveats)

Both source datasets predict `nt` per neuron, not per synapse (Shiu's own classifier, Eckstein
et al. 2024, predicts per presynaptic *site* and takes a >50%-of-sites majority vote per neuron
— see the Methods quote above; MaleCNS/BANC's `predicted_nt` is each dataset's own equivalent
per-neuron aggregate, not re-derivable from the per-site table here, which was out of scope/
budget, see docs/DATA.md), so `sign` here is a neuron-level property either way, matching Shiu.

Caveats (changed 2026-09-18, was: gaba/glutamate -> -1, acetylcholine -> +1, everything else
-> 0 — silenced ~13% of MaleCNS's Traced neurons' *outgoing* edges entirely, which is a much
bigger effect than "unclassified" sounds: `signed_weight = sign(pre) * syn_count`, so
`sign == 0` was not a neutral/unknown marker downstream, it deleted every recurrent synapse
those neurons make, breaking propagation through them. Changed to match Shiu exactly instead):
    - Neurotransmitter identity is a machine-learning prediction from EM synapse morphology and
      ultrastructure (Eckstein et al. 2024 method for MaleCNS; the BANC pipeline predicts per
      neuron the same way) — MaleCNS's own documentation quotes a per-class error rate on the
      order of 6-13% depending on the neurotransmitter; `nt_conf` carries the model's confidence
      but not an external validation figure.
    - Histamine is the photoreceptor transmitter and is typically hyperpolarising/inhibitory via
      histamine-gated chloride channels — biologically the opposite of "excitatory". Shiu's own
      paper does not special-case it (their classifier predicts only 6 classes: ACh/GABA/Glu/DA/
      5-HT/OA, no histamine); MaleCNS and BANC's classifiers add histamine (and BANC adds
      tyramine) as extra predicted classes Shiu never had to decide about. Per this module's
      explicit task ("follow Shiu for the model, note the biology disagrees"), both fall under
      Shiu's "everything else -> excitatory" rule for the *model*, not because that is
      biologically correct.
    - `nt == "unclear"` or null (no confident prediction; ~8.7% of MaleCNS Traced neurons) also
      gets Shiu's default (+1, excitatory) for the same reason: Shiu's classifier always outputs
      one of 6 classes, so "no confident call" cannot arise in Shiu's own pipeline and their
      paper gives no rule for it — "assigned inhibitory only if provably GABA/Glut, otherwise
      excitatory" is the closest reading of a rule written as a strict binary partition.
    - A neuron with `sign == +1` by the null/unclear/histamine/tyramine path is not a validated
      excitatory neuron — it is "not GABA or glutamate", by construction, same caveat Shiu's own
      paper carries for dopaminergic/octopaminergic/serotonergic neurons ("other neurons... will
      be modelled less well").
"""

from __future__ import annotations

import pyarrow as pa

NEURON_SCHEMA = pa.schema(
    [
        pa.field("neuron_id", pa.int64()),
        pa.field("cell_type", pa.string()),
        pa.field("cell_class", pa.string()),
        pa.field("super_class", pa.string()),
        pa.field("side", pa.string()),
        pa.field("nt", pa.string()),
        pa.field("nt_conf", pa.float64()),
        pa.field("sex", pa.string()),
        pa.field("sign", pa.int8()),
    ]
)

EDGE_SCHEMA = pa.schema(
    [
        pa.field("pre_id", pa.int64()),
        pa.field("post_id", pa.int64()),
        pa.field("syn_count", pa.int64()),
    ]
)

# Neurotransmitter -> sign, see the module docstring for the reasoning and caveats. Shiu et al.
# 2024's rule is a strict binary partition (no "unknown" bucket): only gaba/glutamate are
# inhibitory, DEFAULT_SIGN (+1, excitatory) covers everything else, including acetylcholine,
# dopamine/serotonin/octopamine (explicit in Shiu), and tyramine/histamine/unclear/null (not in
# Shiu's own classifier's output, extended by the same rule — see module docstring caveats).
NT_SIGN: dict[str, int] = {
    "gaba": -1,
    "glutamate": -1,
}
DEFAULT_SIGN = 1


def nt_to_sign(nt: str | None) -> int:
    """Map a predicted neurotransmitter name to +1/-1 (Shiu et al.'s binary rule; no 0 case —
    see module docstring)."""
    if nt is None:
        return DEFAULT_SIGN
    return NT_SIGN.get(nt.strip().lower(), DEFAULT_SIGN)
