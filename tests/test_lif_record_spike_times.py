"""`sim.lif.LIFNetwork`'s opt-in `record_spike_times` extension (added for
`bioreservoir.live.frames`) — tiny synthetic graphs only, `codegen_target="numpy"` (no C++
compile, keeps this fast) per this worktree's "no full-brain simulations" rule. Confirms: (1) the
default (`record_spike_times=False`) is unchanged, (2) opting in returns real per-spike
neuron/time arrays consistent with the existing spike-count output.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioreservoir.sim.lif import LIFNetwork

pytestmark = pytest.mark.skipif(
    pytest.importorskip("brian2", reason="brian2 not installed (sim extra)") is None,
    reason="brian2 not installed",
)

N_NEURONS = 40


def _driven_ring_graph():
    """A small ring (each neuron excites the next) so a driven neuron reliably cascades —
    deterministic-enough activity to exercise both monitors without needing a real connectome."""
    pre_idx = np.arange(N_NEURONS, dtype=np.int64)
    post_idx = (pre_idx + 1) % N_NEURONS
    weight = np.full(N_NEURONS, 30.0)  # strongly excitatory, all one presynaptic sign
    return N_NEURONS, pre_idx, post_idx, weight


def test_default_behaviour_is_unchanged_without_record_spike_times():
    n_neurons, pre_idx, post_idx, weight = _driven_ring_graph()
    net = LIFNetwork(n_neurons, pre_idx, post_idx, weight, codegen_target="numpy")
    result = net.run_trial((np.array([0]), np.array([300.0])), duration_ms=50.0, seed=1)
    assert result.spike_neuron_idx is None
    assert result.spike_time_ms is None
    assert result.spike_counts.shape == (N_NEURONS,)


def test_record_spike_times_returns_consistent_per_spike_arrays():
    n_neurons, pre_idx, post_idx, weight = _driven_ring_graph()
    net = LIFNetwork(
        n_neurons, pre_idx, post_idx, weight, codegen_target="numpy", record_spike_times=True
    )
    result = net.run_trial((np.array([0]), np.array([300.0])), duration_ms=50.0, seed=1)
    assert result.spike_neuron_idx is not None
    assert result.spike_time_ms is not None
    assert result.spike_neuron_idx.shape == result.spike_time_ms.shape
    # Every recorded spike time falls inside the trial window.
    assert (result.spike_time_ms >= 0.0).all()
    assert (result.spike_time_ms <= 50.0 + 1e-6).all()
    # Per-neuron spike COUNTS from the two monitors must agree (they observe the same run).
    counted_from_timed = np.bincount(result.spike_neuron_idx, minlength=N_NEURONS)
    assert np.array_equal(counted_from_timed, result.spike_counts)


def test_record_spike_times_is_deterministic_given_seed():
    n_neurons, pre_idx, post_idx, weight = _driven_ring_graph()
    net = LIFNetwork(
        n_neurons, pre_idx, post_idx, weight, codegen_target="numpy", record_spike_times=True
    )
    r1 = net.run_trial((np.array([0]), np.array([300.0])), duration_ms=50.0, seed=7)
    r2 = net.run_trial((np.array([0]), np.array([300.0])), duration_ms=50.0, seed=7)
    assert np.array_equal(r1.spike_neuron_idx, r2.spike_neuron_idx)
    assert np.array_equal(r1.spike_time_ms, r2.spike_time_ms)
