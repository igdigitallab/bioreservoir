"""Bias -> yes/no verdict + confidence, shared by the real brain's own answer (`pipeline.py`) and
every "which one is the real fly?" game contender (`game.py`). Split out of `pipeline.py`
(2026-09-18, the game feature) so the SAME formula produces the real answer's top-level
`confidence` and a control's `decisiveness` bar -- the game's "how hard it turned" comparison must
be apples-to-apples, not two independently-drifting implementations. `pipeline.py` re-exports both
names unchanged, so every existing `pipeline.confidence_from_bias`/`pipeline.answer_yes_no` caller
(including tests) keeps working with no change.
"""

from __future__ import annotations

import json

import numpy as np

# Confidence scale: docs/MODEL.md's calibration puts the male brain's real lateral-bias effect
# sizes at ~0.10-0.16 for a narrow driven population (Johnston's organ) and ~0.03-0.13 for the
# broad `sensory_all` input regime this pipeline actually uses (config.yaml's `input.population`).
# 0.3 is comfortably above every calibrated number, so confidence saturates at 1.0 only for an
# unusually strong reading, not for a typical one -- the live page should look confident rarely,
# not always, which is a deliberate conservative choice, not a calibrated constant.
BIAS_CONFIDENCE_SCALE = 0.3


def confidence_from_bias(corrected_bias: float | None, scale: float = BIAS_CONFIDENCE_SCALE) -> float:
    """Monotonic |bias| -> 0.5..1.0. `None` (every trial zero-spike, or a no-brain reading with
    zero input drive on both sides -- nothing to be confident about) maps to the minimum, 0.5."""
    if corrected_bias is None:
        return 0.5
    return float(np.clip(0.5 + 0.5 * min(abs(corrected_bias) / scale, 1.0), 0.5, 1.0))


def answer_yes_no(p_yes: float) -> str:
    return "yes" if p_yes >= 0.5 else "no"


# -- turn strength (scaling task brief: replaces the 50-100% `confidence` on public surfaces with
# a number a stranger can sanity-check against a real, single, narrow biological stimulus rather
# than an arbitrary 0.3 scale constant) -----------------------------------------------------------

# docs/MODEL.md Sec Calibration, "Chosen readout": the male brain's own single-antenna Johnston's-
# organ (JO) calibration -- JO left driven -> descending_all bias +0.118 +/- 0.002, JO right driven
# -> -0.157 +/- 0.005 (5 trials/side, `python -m bioreservoir.sim.calibrate`). This constant is
# ONLY the fallback used if the calibration JSON below is missing or its schema has moved out from
# under `_load_turn_strength_ref` -- the normal path reads the two numbers straight out of the
# committed file so a re-calibration updates this automatically, no code change required.
TURN_STRENGTH_REF_FALLBACK = 0.1375  # mean(|0.118|, |0.157|) -- see docs/MODEL.md Sec Calibration


def _load_turn_strength_ref() -> float:
    """The mean |lateral bias| of the male brain's own single-antenna JO calibration, read from
    the same committed calibration JSON `validation.calibration_summary` already quotes verbatim
    (`lateral_bias.jo_{left,right}_driven.descending_bias.mean` -- the exact key path
    `validation.lateral_validation_verdict` already relies on, so it is as "stable" as this
    codebase's own established convention for these two numbers gets). Any failure (no calibration
    file committed yet, an unexpected shape) falls back to `TURN_STRENGTH_REF_FALLBACK` rather than
    raising -- `import bioreservoir.live.verdict` must never fail just because a calibration run
    has not landed yet, same convention `config.py`'s module docstring states for env vars."""
    from bioreservoir.live import config

    try:
        calibration_dir = config.EXPERIMENT_DIR / "calibration"
        candidates = sorted(calibration_dir.glob("*-malecns.json"))  # YYYY-MM-DD- prefix sorts chronologically
        if not candidates:
            return TURN_STRENGTH_REF_FALLBACK
        raw = json.loads(candidates[-1].read_text())
        lateral = raw["lateral_bias"]
        left = lateral["jo_left_driven"]["descending_bias"]["mean"]
        right = lateral["jo_right_driven"]["descending_bias"]["mean"]
        return (abs(float(left)) + abs(float(right))) / 2.0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return TURN_STRENGTH_REF_FALLBACK


# Computed once at import (same convention as `config.py`'s module-level env-var reads) -- every
# caller shares one value for the life of the process; a fresh calibration run only takes effect
# after a restart, which matches how every other calibration-derived number in this codebase
# (confidence's own `BIAS_CONFIDENCE_SCALE`, the handedness `b0` cache) already behaves.
TURN_STRENGTH_REF = _load_turn_strength_ref()


def turn_strength_from_bias(corrected_bias: float | None, ref: float = TURN_STRENGTH_REF) -> float:
    """`clip(|corrected_bias| / ref, 0, 1)` (task brief) -- unlike `confidence_from_bias`'s 0.5..1.0
    scale (built to never look unconfident), this is a plain 0..1 fraction of a real, named
    biological effect size, meant to read as "barely more than noise" for a typical question."""
    if corrected_bias is None:
        return 0.0
    return float(np.clip(abs(corrected_bias) / ref, 0.0, 1.0))


def answer_turn_strength(answer: dict) -> float:
    """`turn_strength` for a full Answer dict, new or old: `pipeline.compute_answer` now stores it
    directly, but a row answered before that field existed has none -- computed here on the fly
    from the SAME handedness-corrected bias the stored `answer/lateral_bias` already carries (task
    brief: "compute it on the fly for old rows that lack it"), never re-derived from anything the
    stored row does not already have."""
    stored = answer.get("turn_strength")
    if stored is not None:
        return float(stored)
    return turn_strength_from_bias(answer.get("lateral_bias"))


def with_turn_strength(answer: dict) -> dict:
    """`answer`, guaranteed to carry a `turn_strength` key -- a shallow-copied dict when one had to
    be computed (never mutates the caller's copy, which may be a value straight out of `LiveStore`
    that other code still holds a reference to), the SAME object when it already had one."""
    if answer.get("turn_strength") is not None:
        return answer
    return {**answer, "turn_strength": answer_turn_strength(answer)}
