"""Handedness correction (README.md "Handedness and side mapping").

A 2026-09-18 supervised smoke run (male brain, one test question) found a real-connectome lateral
bias of about -0.10 for both the question's `original` and `negation` phrasing — the *same* sign
and similar magnitude regardless of what was asked, while the rewired and no-brain controls showed
a much smaller, opposite-sign bias. That is the brain's own intrinsic left/right asymmetry
("handedness"), not a question-driven signal, and it would have read as a uniform partisan skew
across the 33 "will <incumbent> hold" election questions (memory `fly-handedness.md`).

`b0` is measured once per `(brain, condition)` — real, rewired, er, and the no-brain baseline —
from `oracle.reference`'s 24 neutral sentences (mean lateral bias over every sentence and every one
of its trials, each sentence's own trials already averaged by
`oracle.readout.aggregate_trials`/the no-brain control's own single bias value). Every question's
raw bias then has that group's `b0` subtracted before `oracle.readout.probability_from_bias` turns
it into a probability, using `oracle.side_mapping`'s per-question YES-side coin instead of a global
flag.
"""

from __future__ import annotations

import numpy as np


def compute_b0(mean_biases: list[float]) -> float:
    """`b0 = mean(mean_biases)` — the group's own handedness, over every reference sentence's
    already-per-sentence-averaged bias. Raises if `mean_biases` is empty: a `(brain, condition)`
    with no reference runs yet has no defined handedness, and silently returning 0.0 would let an
    unmeasured group's questions through uncorrected without anyone noticing."""
    if not mean_biases:
        raise ValueError("compute_b0: no reference biases given — reference runs not done yet")
    return float(np.mean(mean_biases))


def corrected_bias(raw_bias: float, b0: float) -> float:
    """A question's raw lateral bias with its `(brain, condition)` group's own handedness
    subtracted out."""
    return raw_bias - b0


def apply_correction(
    raw_bias: float | None,
    b0: float,
    left_is_yes: bool,
    zero_spike_probability: float,
) -> tuple[float, float | None]:
    """`(p_yes, corrected_bias)` for one question run. `raw_bias` is `None` when there is nothing
    to correct — every trial was zero-spike (`oracle.readout.ReadoutResult.all_trials_zero_spikes`)
    or, for the no-brain control, the input drive itself summed to zero on both sides
    (`oracle.controls.no_brain_baseline`'s own `bias is None` case) — in which case handedness
    correction is skipped entirely and `p_yes` falls back to `zero_spike_probability`, exactly as
    it did before this correction existed. Otherwise `b0` is subtracted and the result is mapped to
    a probability via `oracle.readout.probability_from_bias` using this question's own
    `left_is_yes` (`oracle.side_mapping.left_is_yes_for`), not a global flag.
    """
    from bioreservoir.oracle.readout import probability_from_bias

    if raw_bias is None:
        return zero_spike_probability, None
    corrected = corrected_bias(raw_bias, b0)
    return probability_from_bias(corrected, left_is_yes=left_is_yes), corrected


def load_b0(ledger, brain: str, condition: str, config_hash: str) -> float:
    """`b0` for `(brain, condition)` under `config_hash`, read from the ledger's `done` reference
    runs (`Ledger.reference_mean_biases`) — the single source of truth so a resumed run picks up
    exactly the same `b0` a from-scratch run would, whether or not this call itself just finished
    running any reference items."""
    biases = ledger.reference_mean_biases(brain, condition, config_hash)
    return compute_b0(biases)
