"""Behavioural-state readouts — the page calls them the fly's "emotions" — spike counts from
named populations, normalized to 0..1, published only when `experiments/001-fly-oracle/
live-states.yaml` marks that state `validated: true` (see that file's header for why, and
`validate_states.py` for how a state gets validated).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from bioreservoir.live import config

STATE_NAMES = ("appetite", "fear", "backoff", "courtship", "arousal")


@dataclass(frozen=True)
class StateSpec:
    name: str
    populations: tuple[str, ...]
    validated: bool
    normalization: dict | None
    note: str | None


def load_states(path: Path = config.LIVE_STATES_YAML) -> dict[str, StateSpec]:
    raw = yaml.safe_load(path.read_text())
    specs: dict[str, StateSpec] = {}
    for name, spec in raw["states"].items():
        specs[name] = StateSpec(
            name=name,
            populations=tuple(spec.get("populations", ())),
            validated=bool(spec["validated"]),
            normalization=spec.get("normalization"),
            note=spec.get("note"),
        )
    return specs


def raw_appetite_fear_backoff_courtship(
    spike_counts: dict[str, int], spec: StateSpec
) -> int:
    """Summed spike count across a state's named populations (`spike_counts` is
    `{population_name: total_spikes_over_all_trials}`, populations not present in this brain's
    populations.yaml — or absent for this dataset — are simply skipped)."""
    return sum(spike_counts.get(pop, 0) for pop in spec.populations)


def raw_arousal(n_active_neurons: int, n_total_neurons: int) -> float:
    """Fraction of the whole network that spiked at least once, averaged the same way every
    other per-trial number here is (caller passes already-trial-averaged counts, see
    `pipeline.py`)."""
    if n_total_neurons == 0:
        return 0.0
    return n_active_neurons / n_total_neurons


def normalize(spec: StateSpec, raw_value: float) -> float | None:
    """`None` if `spec.validated` is `False` (task brief: "published (non-null) only if marked
    validated"). Otherwise `clip((raw - min) / (max - min), 0, 1)` using whichever
    `normalization` key pair is present (`{min_spikes, max_spikes}` for the four spike-count
    states, `{min_frac, max_frac}` for `arousal`)."""
    if not spec.validated:
        return None
    if spec.normalization is None:
        raise ValueError(f"state {spec.name!r} is validated but has no normalization range")
    keys = sorted(spec.normalization.keys())
    lo_key = next(k for k in keys if k.startswith("min_"))
    hi_key = next(k for k in keys if k.startswith("max_"))
    lo, hi = spec.normalization[lo_key], spec.normalization[hi_key]
    if hi <= lo:
        raise ValueError(f"state {spec.name!r} has a degenerate normalization range: {spec.normalization}")
    return float(np.clip((raw_value - lo) / (hi - lo), 0.0, 1.0))


def compute_states(
    specs: dict[str, StateSpec],
    spike_counts: dict[str, int],
    n_active_neurons: int,
    n_total_neurons: int,
) -> dict[str, float | None]:
    """The Answer schema's `states` dict: one 0..1-or-null value per `STATE_NAMES`."""
    result: dict[str, float | None] = {}
    for name in STATE_NAMES:
        spec = specs[name]
        if name == "arousal":
            raw = raw_arousal(n_active_neurons, n_total_neurons)
        else:
            raw = raw_appetite_fear_backoff_courtship(spike_counts, spec)
        result[name] = normalize(spec, raw)
    return result
