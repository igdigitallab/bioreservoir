"""Per-25ms-bin active-neuron-index encoding for the Answer's `frames` field — which neurons
fired, bin by bin, for the page's spike replay. Pure numpy/base64 — no Brian2 import, so this is fully unit-testable with
small synthetic spike arrays (`sim/lif.py`'s `record_spike_times=True` extension is what actually
produces `spike_neuron_idx`/`spike_time_ms` at runtime).
"""

from __future__ import annotations

import base64

import numpy as np

from bioreservoir.live import config


def bin_index(spike_time_ms: np.ndarray, duration_ms: float, bin_ms: float) -> np.ndarray:
    """Which bin (0-based) each spike time falls into; clipped to the last bin so a spike at
    exactly `duration_ms` (Brian2's own half-open `[0, duration)` window makes this rare, but
    floating-point boundary spikes are cheap to guard against) does not index out of range."""
    n_bins = n_bins_for(duration_ms, bin_ms)
    idx = np.floor(np.asarray(spike_time_ms, dtype=np.float64) / bin_ms).astype(np.int64)
    return np.clip(idx, 0, n_bins - 1)


def n_bins_for(duration_ms: float, bin_ms: float) -> int:
    return int(np.ceil(duration_ms / bin_ms))


def active_dense_indices_per_bin(
    spike_neuron_idx: np.ndarray, spike_time_ms: np.ndarray, duration_ms: float, bin_ms: float
) -> list[np.ndarray]:
    """One sorted, deduplicated array of dense (Brian2) neuron indices per bin — a neuron that
    spiked more than once in the same bin appears there once (the frontend only needs "did this
    soma light up this bin", not a count)."""
    n_bins = n_bins_for(duration_ms, bin_ms)
    if len(spike_neuron_idx) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n_bins)]
    bins = bin_index(spike_time_ms, duration_ms, bin_ms)
    result = []
    neuron_idx = np.asarray(spike_neuron_idx, dtype=np.int64)
    for b in range(n_bins):
        result.append(np.unique(neuron_idx[bins == b]))
    return result


def map_to_atlas(dense_indices: np.ndarray, dense_to_atlas: dict[int, int]) -> np.ndarray:
    """Dense (Brian2) indices -> atlas indices, dropping any dense index the atlas doesn't cover
    (`atlas.build_dense_to_atlas`'s own docstring: absence there is not an error)."""
    if dense_indices.size == 0:
        return np.empty(0, dtype=np.uint32)
    mapped = [dense_to_atlas[i] for i in dense_indices.tolist() if i in dense_to_atlas]
    return np.array(mapped, dtype=np.uint32)


def cap_bin(
    atlas_indices: np.ndarray, cap: int, seed: int
) -> tuple[np.ndarray, bool]:
    """Seeded subsample down to `cap` entries if over (task brief: "Cap each bin to 6,000
    indices (seeded subsample, flag it)"). `seed` should be unique per (question, trial, bin) so
    two different over-cap bins do not always drop the exact same relative subset."""
    if atlas_indices.size <= cap:
        return atlas_indices, False
    rng = np.random.default_rng(seed)
    chosen = rng.choice(atlas_indices, size=cap, replace=False)
    chosen.sort()
    return chosen, True


def encode_bin_b64(atlas_indices: np.ndarray) -> str:
    """Little-endian `uint32` array -> base64 (Answer schema: `frames.active_b64[i]`)."""
    arr = np.asarray(atlas_indices, dtype="<u4")
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_bin_b64(encoded: str) -> np.ndarray:
    """Inverse of `encode_bin_b64` — used by tests, not by the live pipeline itself."""
    return np.frombuffer(base64.b64decode(encoded), dtype="<u4")


def frames_payload(
    spike_neuron_idx: np.ndarray,
    spike_time_ms: np.ndarray,
    duration_ms: float,
    dense_to_atlas: dict[int, int],
    bin_ms: float = config.FRAMES_BIN_MS,
    cap: int = config.FRAMES_MAX_ACTIVE_PER_BIN,
    seed: int = 0,
) -> dict:
    """Full Answer `frames` field: `{"bin_ms", "n_bins", "active_b64", "capped"}`. `capped` is an
    additive field beyond the spec's documented three keys (task brief: "flag it") — safe for the
    frontend to ignore, not part of the documented required set."""
    bins = active_dense_indices_per_bin(spike_neuron_idx, spike_time_ms, duration_ms, bin_ms)
    active_b64 = []
    any_capped = False
    for i, dense_bin in enumerate(bins):
        atlas_bin = map_to_atlas(dense_bin, dense_to_atlas)
        atlas_bin, capped = cap_bin(atlas_bin, cap, seed=seed + i)
        any_capped = any_capped or capped
        active_b64.append(encode_bin_b64(atlas_bin))
    return {
        "bin_ms": bin_ms,
        "n_bins": len(bins),
        "active_b64": active_b64,
        "capped": any_capped,
    }
