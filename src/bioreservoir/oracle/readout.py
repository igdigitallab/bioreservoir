"""Spike counts -> P(yes) (README.md Pipeline step 3).

The readout is a lateral bias between a left and a right population, named by config
(`experiments/001-fly-oracle/config.yaml`'s `readout.name`, default `descending_all` ->
`descending_all_left`/`descending_all_right` in populations.yaml — see docs/MODEL.md for why this,
not the smaller DNa01/DNa02/MN9/DNp01 pairs README.md originally sketched, is the calibrated
primary readout). `oracle.controls.no_brain_baseline` reuses `probability_from_bias` directly on
the *input* drive's left/right rates, so the bias-to-probability mapping is defined once here and
shared by both the brain readout and its no-brain control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

P_YES_MIN = 0.01
P_YES_MAX = 0.99
BIAS_EPS = 1e-6


@dataclass(frozen=True)
class TrialReadout:
    left: float
    right: float
    bias: float | None  # None if left+right == 0 (zero-spike trial, see `lateral_bias`)
    zero_spikes: bool


@dataclass(frozen=True)
class ReadoutResult:
    p_yes: float
    mean_bias: float
    n_trials: int
    n_zero_spike_trials: int
    all_trials_zero_spikes: bool
    trials: list[TrialReadout]


def lateral_bias(left: float, right: float, eps: float = BIAS_EPS) -> float | None:
    """`(L - R) / (L + R + eps)`, or `None` if both sides are exactly 0 (a trial with no spikes
    on either the left or right readout population — the ratio is undefined, not "no bias";
    `epsilon` only guards against one-sided near-zero denominators, not the fully-silent case,
    which callers must handle explicitly per the task brief)."""
    if left == 0.0 and right == 0.0:
        return None
    return (left - right) / (left + right + eps)


def probability_from_bias(bias: float, left_is_yes: bool = True, p_min: float = P_YES_MIN, p_max: float = P_YES_MAX) -> float:
    """`P(yes) = clip((1 + b) / 2, p_min, p_max)`, or `clip((1 - b) / 2, ...)` if `left_is_yes` is
    False (the yes/no-swap invariance variant, README.md "Invariance tests" — flips the mapping,
    not the readout population or which side is anatomically "left")."""
    raw = (1.0 + bias) / 2.0 if left_is_yes else (1.0 - bias) / 2.0
    return float(np.clip(raw, p_min, p_max))


def trial_readout(left: float, right: float) -> TrialReadout:
    """One trial's raw readout: left/right spike sums and their bias (`None` if both are 0).

    Per-trial `P(yes)` is intentionally not computed here — the task's readout rule is "averaged
    over trials -> P(yes)" (`aggregate_trials`), not "P(yes) per trial, then averaged"; those two
    orders give different numbers once `clip()` is in the mix, and only the first is what
    README.md's readout step describes.
    """
    bias = lateral_bias(left, right)
    return TrialReadout(left=left, right=right, bias=bias, zero_spikes=bias is None)


def aggregate_trials(
    trials: list[TrialReadout], left_is_yes: bool, zero_spike_probability: float
) -> ReadoutResult:
    """`b = mean(bias over trials) -> P(yes)` (task brief: "averaged over trials"). Zero-spike
    trials contribute a neutral bias of 0.0 to the mean (their `TrialReadout.bias` is already
    `None`, not a fabricated non-zero number) and are counted separately so a report can flag
    "N of n_trials had no spikes on either side" without hiding it inside the average. If every
    trial was zero-spike, the mean bias is 0.0 and `P(yes)` falls back to
    `zero_spike_probability`, and `all_trials_zero_spikes` is set so callers can flag the whole
    result, not just individual trials.
    """
    n = len(trials)
    n_zero = sum(1 for t in trials if t.zero_spikes)
    biases = [t.bias if t.bias is not None else 0.0 for t in trials]
    mean_bias = float(np.mean(biases)) if biases else 0.0
    all_zero = n > 0 and n_zero == n
    p_yes = (
        zero_spike_probability
        if all_zero
        else probability_from_bias(mean_bias, left_is_yes=left_is_yes)
    )
    return ReadoutResult(
        p_yes=p_yes,
        mean_bias=mean_bias,
        n_trials=n,
        n_zero_spike_trials=n_zero,
        all_trials_zero_spikes=all_zero,
        trials=trials,
    )
