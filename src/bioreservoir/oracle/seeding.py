"""Deterministic seed derivation shared by encode.py, controls.py, ledger.py and plan.py.

Every seed anywhere in the pipeline is derived from a small set of base seeds
(`experiments/001-fly-oracle/config.yaml`'s `seed:` block) plus a stable, process- and
platform-independent string hash — never Python's built-in `hash()`, which is salted per-process
(`PYTHONHASHSEED`) unless explicitly fixed, and would silently break reproducibility (the same
caution `sim/calibrate.py`'s `input_regime_scan` docstring already states for its own grid seeds).
"""

from __future__ import annotations

import hashlib


def stable_seed(*parts: str, base: int = 0) -> int:
    """A deterministic 63-bit non-negative seed derived from `base` and the string `parts`.

    `numpy.random.default_rng` accepts any non-negative int, so 63 bits (not the full 64) avoids
    any sign-bit ambiguity when this value is later combined with other seeds via addition.
    """
    key = "\x1f".join((str(base), *parts))  # \x1f (unit separator) as a field delimiter that
    # cannot appear in question ids/variant names, so "a", "b" and "ab" cannot collide.
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def trial_seed(base: int, question_id: str, brain: str, condition: str, variant: str, trial_index: int) -> int:
    """Seed for one `LIFNetwork.run_trial` call, unique per (question, brain, condition, variant,
    trial index) so every trial in the whole batch draws an independent Poisson realization, and
    reproducible: re-running the same combination always regenerates the same seed.

    Reduced to 32 bits: Brian2's `seed()` forwards to `numpy.random.seed`, which only accepts
    0..2**32-1."""
    return stable_seed(question_id, brain, condition, variant, str(trial_index), base=base) % 2**32


def reference_trial_seed(base: int, reference_id: str, brain: str, condition: str, trial_index: int) -> int:
    """Seed for one handedness-reference `LIFNetwork.run_trial` call (`oracle.reference`,
    `oracle.handedness`) — same construction as `trial_seed`, but reference sentences have no text
    variant, and the leading `"ref"` literal is an extra guard against a reference sentence id ever
    colliding with a question id in the seed derivation (the primary guard is the two files' own
    disjoint id conventions, `"ref-NN"` vs `questions.yaml`'s `"2026-..."` ids)."""
    return stable_seed("ref", reference_id, brain, condition, str(trial_index), base=base) % 2**32
