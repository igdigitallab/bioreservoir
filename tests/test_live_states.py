"""states.py: live-states.yaml loading + spike-count -> 0..1 normalization."""

from __future__ import annotations

import pytest

from bioreservoir.live import states


def test_load_states_reads_the_committed_live_states_yaml():
    specs = states.load_states()
    assert set(specs.keys()) == set(states.STATE_NAMES)
    assert specs["arousal"].validated is True
    for name in ("appetite", "fear", "backoff", "courtship"):
        assert specs[name].validated is False


def test_committed_courtship_state_names_both_populations():
    specs = states.load_states()
    assert specs["courtship"].populations == ("P1_pC1", "pIP10")


def test_arousal_has_a_normalization_range_by_default():
    specs = states.load_states()
    assert specs["arousal"].normalization == {"min_frac": 0.0, "max_frac": 1.0}


def test_raw_appetite_fear_backoff_courtship_sums_named_populations():
    spec = states.StateSpec(
        name="courtship", populations=("P1_pC1", "pIP10"), validated=False, normalization=None, note=None
    )
    raw = states.raw_appetite_fear_backoff_courtship({"P1_pC1": 40, "pIP10": 5, "other": 999}, spec)
    assert raw == 45


def test_raw_appetite_fear_backoff_courtship_ignores_missing_populations():
    spec = states.StateSpec(name="fear", populations=("DNp01_giant_fiber",), validated=False, normalization=None, note=None)
    assert states.raw_appetite_fear_backoff_courtship({}, spec) == 0


def test_raw_arousal_is_a_fraction():
    assert states.raw_arousal(n_active_neurons=50, n_total_neurons=200) == pytest.approx(0.25)
    assert states.raw_arousal(n_active_neurons=0, n_total_neurons=0) == 0.0


def test_normalize_returns_none_when_not_validated():
    spec = states.StateSpec(name="fear", populations=(), validated=False, normalization=None, note=None)
    assert states.normalize(spec, 500.0) is None


def test_normalize_clips_into_0_1_using_min_max_spikes():
    spec = states.StateSpec(
        name="fear", populations=(), validated=True,
        normalization={"min_spikes": 10.0, "max_spikes": 110.0}, note=None,
    )
    assert states.normalize(spec, 10.0) == pytest.approx(0.0)
    assert states.normalize(spec, 60.0) == pytest.approx(0.5)
    assert states.normalize(spec, 110.0) == pytest.approx(1.0)
    assert states.normalize(spec, 1000.0) == pytest.approx(1.0)  # clipped
    assert states.normalize(spec, -50.0) == pytest.approx(0.0)  # clipped


def test_normalize_raises_on_validated_without_normalization():
    spec = states.StateSpec(name="fear", populations=(), validated=True, normalization=None, note=None)
    with pytest.raises(ValueError):
        states.normalize(spec, 10.0)


def test_normalize_raises_on_degenerate_range():
    spec = states.StateSpec(
        name="fear", populations=(), validated=True,
        normalization={"min_spikes": 10.0, "max_spikes": 10.0}, note=None,
    )
    with pytest.raises(ValueError):
        states.normalize(spec, 10.0)


def test_compute_states_publishes_null_for_unvalidated_states_and_a_value_for_arousal():
    specs = states.load_states()
    result = states.compute_states(specs, spike_counts={"MN9_proboscis": 50}, n_active_neurons=100, n_total_neurons=1000)
    assert set(result.keys()) == set(states.STATE_NAMES)
    for name in ("appetite", "fear", "backoff", "courtship"):
        assert result[name] is None
    assert result["arousal"] == pytest.approx(0.1)
