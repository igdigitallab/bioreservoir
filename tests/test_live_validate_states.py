"""validate_states.py: the pure statistics (`propose_validation`) with synthetic numbers, plus
`run_state_readings`/`load_varied` against the real MaleCNS graph + a `FakeNet` (same convention
as `tests/test_live_worker.py` — no Brian2, no full-brain simulation).
"""

from __future__ import annotations

import numpy as np
import pytest

from bioreservoir.live import config, states, validate_states
from bioreservoir.oracle import config as oracle_config


def test_load_varied_reads_the_committed_24_question_file():
    varied = validate_states.load_varied()
    assert len(varied) == 24
    assert all(v.text.strip().endswith("?") for v in varied)


def test_load_varied_raises_a_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_states.load_varied(tmp_path / "nope.yaml")


# -- propose_validation: pure stats, no simulation at all -----------------------------------------


def _reading(text_id, trial_values):
    return validate_states.TextStateReading(text_id=text_id, trial_values=trial_values)


def test_propose_validation_true_when_across_text_variance_dwarfs_trial_noise():
    # Each text has near-zero trial-to-trial noise, but means vary a lot text-to-text.
    readings = [_reading(f"t{i}", [10.0 * i, 10.0 * i + 0.1, 10.0 * i - 0.1]) for i in range(10)]
    validated, normalization, stats = validate_states.propose_validation(readings, threshold=2.0)
    assert validated is True
    assert stats["across_text_std"] > stats["mean_within_text_std"]
    assert normalization["min_spikes"] < normalization["max_spikes"]


def test_propose_validation_false_when_it_is_all_just_trial_noise():
    rng = np.random.default_rng(0)
    # Every text has the SAME underlying mean (50), only noisy trial-to-trial variation.
    readings = [_reading(f"t{i}", (50 + rng.normal(scale=10, size=5)).tolist()) for i in range(10)]
    validated, _normalization, _stats = validate_states.propose_validation(readings, threshold=2.0)
    assert validated is False


def test_propose_validation_handles_all_zero_readings_without_crashing():
    readings = [_reading(f"t{i}", [0.0, 0.0, 0.0]) for i in range(5)]
    validated, normalization, _stats = validate_states.propose_validation(readings)
    assert validated is False
    assert normalization["min_spikes"] == 0.0


def test_write_proposals_updates_only_the_given_states_never_arousal(tmp_path):
    import shutil

    dest = tmp_path / "live-states.yaml"
    shutil.copy(config.LIVE_STATES_YAML, dest)
    before = states.load_states(dest)
    assert before["fear"].validated is False

    proposals = {"fear": (True, {"min_spikes": 1.0, "max_spikes": 99.0}, {})}
    validate_states.write_proposals(dest, proposals)

    after = states.load_states(dest)
    assert after["fear"].validated is True
    assert after["fear"].normalization == {"min_spikes": 1.0, "max_spikes": 99.0}
    # untouched states keep their committed values.
    assert after["courtship"].validated is False
    assert after["arousal"].validated is True
    assert after["arousal"].normalization == {"min_frac": 0.0, "max_frac": 1.0}


# -- run_state_readings against the real graph + a deterministic FakeNet -------------------------


@pytest.fixture(scope="module")
def cfg():
    return oracle_config.load_config()


@pytest.fixture(scope="module")
def graph(cfg):
    from bioreservoir.sim import bench

    _n, _pre, _post, _w, id_to_dense = bench.graph_to_arrays("malecns", min_syn=cfg.trial.min_syn)
    return id_to_dense


@pytest.fixture(scope="module")
def state_specs():
    return states.load_states()


@pytest.fixture(scope="module")
def state_population_idx(state_specs, graph):
    from bioreservoir.sim import bench

    idx: dict[str, np.ndarray | None] = {}
    for spec in state_specs.values():
        for pop in spec.populations:
            if pop not in idx:
                idx[pop] = bench.population_indices("malecns", graph, pop)
    return idx


@pytest.fixture(scope="module")
def encoder_model():
    from bioreservoir.oracle.encode import load_encoder

    return load_encoder()


class _FakeNet:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.calls = 0

    def run_trial(self, inputs, duration_ms, seed):
        from types import SimpleNamespace

        self.calls += 1
        rng = np.random.default_rng(seed)
        counts = (rng.random(self.n_neurons) < 0.05).astype(np.int64) * 3
        return SimpleNamespace(spike_counts=counts, spike_neuron_idx=None, spike_time_ms=None)


def test_run_state_readings_covers_every_text_and_every_state(cfg, graph, state_specs, state_population_idx, encoder_model):
    from bioreservoir.oracle import reference

    texts = list(reference.load_reference(cfg.handedness.reference_path()))[:3]  # keep this test fast
    net = _FakeNet(len(graph))
    readings = validate_states.run_state_readings(net, graph, cfg, texts, state_specs, state_population_idx, encoder_model)
    assert set(readings.keys()) == set(states.STATE_NAMES)
    for name in states.STATE_NAMES:
        assert len(readings[name]) == len(texts)
        for r in readings[name]:
            assert len(r.trial_values) == cfg.trial.n_trials
    assert net.calls == len(texts) * cfg.trial.n_trials


def test_run_state_readings_arousal_is_always_between_0_and_1(cfg, graph, state_specs, state_population_idx, encoder_model):
    from bioreservoir.oracle import reference

    texts = list(reference.load_reference(cfg.handedness.reference_path()))[:2]
    net = _FakeNet(len(graph))
    readings = validate_states.run_state_readings(net, graph, cfg, texts, state_specs, state_population_idx, encoder_model)
    for r in readings["arousal"]:
        assert all(0.0 <= v <= 1.0 for v in r.trial_values)
