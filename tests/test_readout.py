from bioreservoir.oracle.readout import (
    aggregate_trials,
    lateral_bias,
    probability_from_bias,
    trial_readout,
)


def test_lateral_bias_symmetric_split_is_zero():
    assert lateral_bias(10, 10) == 0.0


def test_lateral_bias_all_left_is_near_one():
    b = lateral_bias(10, 0)
    assert b is not None
    assert 0.99 < b <= 1.0


def test_lateral_bias_zero_spikes_is_none():
    assert lateral_bias(0, 0) is None


def test_probability_from_bias_clips_to_range():
    assert probability_from_bias(1.0) == 0.99
    assert probability_from_bias(-1.0) == 0.01
    assert probability_from_bias(0.0) == 0.5


def test_probability_from_bias_left_is_yes_flip_is_exact_complement():
    for b in (-0.8, -0.3, 0.0, 0.3, 0.8):
        p_left_yes = probability_from_bias(b, left_is_yes=True)
        p_right_yes = probability_from_bias(b, left_is_yes=False)
        assert abs(p_left_yes - (1.0 - p_right_yes)) < 1e-12


def test_trial_readout_zero_spikes_flagged():
    t = trial_readout(0, 0)
    assert t.zero_spikes is True
    assert t.bias is None

    t2 = trial_readout(3, 1)
    assert t2.zero_spikes is False
    assert abs(t2.bias - 0.5) < 1e-5


def test_aggregate_trials_averages_bias_then_maps_to_probability():
    trials = [trial_readout(10, 0), trial_readout(0, 10)]  # bias +1, bias -1 -> mean bias 0
    result = aggregate_trials(trials, left_is_yes=True, zero_spike_probability=0.5)
    assert abs(result.mean_bias) < 1e-9
    assert abs(result.p_yes - 0.5) < 1e-9
    assert result.n_zero_spike_trials == 0


def test_aggregate_trials_all_zero_spikes_falls_back_to_configured_probability():
    trials = [trial_readout(0, 0), trial_readout(0, 0)]
    result = aggregate_trials(trials, left_is_yes=True, zero_spike_probability=0.37)
    assert result.all_trials_zero_spikes is True
    assert result.n_zero_spike_trials == 2
    assert result.p_yes == 0.37


def test_aggregate_trials_partial_zero_spikes_only_counted_not_hidden():
    trials = [trial_readout(0, 0), trial_readout(10, 0)]
    result = aggregate_trials(trials, left_is_yes=True, zero_spike_probability=0.5)
    assert result.n_zero_spike_trials == 1
    assert result.all_trials_zero_spikes is False
    # zero-spike trial contributes bias=0.0 to the mean, non-zero trial contributes ~1.0
    assert 0.4 < result.mean_bias < 0.6
