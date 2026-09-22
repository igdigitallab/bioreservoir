import pytest

from bioreservoir.oracle import handedness
from bioreservoir.oracle.ledger import Ledger


def test_compute_b0_is_the_mean_of_the_reference_biases():
    assert handedness.compute_b0([-0.1, -0.1, -0.1]) == pytest.approx(-0.1)
    assert handedness.compute_b0([-0.2, 0.0, 0.2]) == pytest.approx(0.0)
    assert handedness.compute_b0([0.05]) == pytest.approx(0.05)


def test_compute_b0_raises_on_empty_input():
    with pytest.raises(ValueError):
        handedness.compute_b0([])


def test_corrected_bias_is_plain_subtraction():
    assert handedness.corrected_bias(raw_bias=-0.095, b0=-0.099) == pytest.approx(0.004)
    assert handedness.corrected_bias(raw_bias=0.3, b0=0.0) == pytest.approx(0.3)


def test_apply_correction_subtracts_b0_before_mapping_to_probability():
    # Raw bias alone would look YES-favoring under the old, uncorrected convention (positive,
    # left_is_yes=True): P(yes) = (1 + 0.2) / 2 = 0.6. Once the group's own handedness (+0.2, i.e.
    # every reference sentence -- which carries no real signal -- already reads +0.2 on this brain)
    # is subtracted, the corrected bias is 0.0 and the honest P(yes) is 0.5.
    p_yes, corrected = handedness.apply_correction(raw_bias=0.2, b0=0.2, left_is_yes=True, zero_spike_probability=0.5)
    assert corrected == pytest.approx(0.0)
    assert p_yes == pytest.approx(0.5)


def test_apply_correction_can_flip_the_sign_of_the_effective_signal():
    # A brain with a strong rightward handedness (b0 = -0.3) that then shows a small further
    # rightward bias on a question (raw -0.35) is actually *less* right-leaning than its own
    # baseline once corrected: corrected = -0.35 - (-0.3) = -0.05, still negative but much smaller.
    p_yes, corrected = handedness.apply_correction(raw_bias=-0.35, b0=-0.3, left_is_yes=True, zero_spike_probability=0.5)
    assert corrected == pytest.approx(-0.05)
    assert p_yes == pytest.approx(0.475)


def test_apply_correction_respects_left_is_yes_mapping():
    p_yes_left, corrected_left = handedness.apply_correction(0.4, b0=0.0, left_is_yes=True, zero_spike_probability=0.5)
    p_yes_right, corrected_right = handedness.apply_correction(0.4, b0=0.0, left_is_yes=False, zero_spike_probability=0.5)
    assert corrected_left == corrected_right == pytest.approx(0.4)
    assert p_yes_left > 0.5
    assert p_yes_right < 0.5
    assert p_yes_left == pytest.approx(1.0 - p_yes_right)


def test_apply_correction_skips_correction_on_none_raw_bias():
    """`raw_bias=None` (all trials zero-spike, or a zero-rate no-brain input) bypasses handedness
    correction entirely and falls back to `zero_spike_probability`, exactly as before this
    correction existed -- there is nothing to subtract b0 from."""
    p_yes, corrected = handedness.apply_correction(raw_bias=None, b0=0.5, left_is_yes=True, zero_spike_probability=0.37)
    assert corrected is None
    assert p_yes == 0.37


def test_load_b0_reads_from_the_ledger_reference_predictions(tmp_path):
    lg = Ledger(path=tmp_path / "ledger.sqlite")
    lg.upsert_reference_sentence("ref-01", "The kettle is on the stove.")
    lg.upsert_reference_sentence("ref-02", "A cat is sleeping on the windowsill.")

    for ref_id, bias in [("ref-01", -0.08), ("ref-02", -0.12)]:
        run_id = lg.start_reference_run(ref_id, "malecns", "real", "cfg1", trial_seeds=[1, 2])
        lg.record_reference_prediction(run_id, mean_bias=bias, n_trials=2, n_zero_spike_trials=0, per_trial_stats=[])
        lg.finish_reference_run(run_id, "done")

    b0 = handedness.load_b0(lg, "malecns", "real", "cfg1")
    assert b0 == pytest.approx(-0.10)
    lg.close()


def test_load_b0_raises_when_no_reference_runs_are_done(tmp_path):
    lg = Ledger(path=tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError):
        handedness.load_b0(lg, "malecns", "real", "cfg1")
    lg.close()


def test_load_b0_only_counts_done_runs_under_the_matching_config_hash(tmp_path):
    lg = Ledger(path=tmp_path / "ledger.sqlite")
    lg.upsert_reference_sentence("ref-01", "The kettle is on the stove.")

    run_id = lg.start_reference_run("ref-01", "malecns", "real", "cfg1", trial_seeds=[1])
    lg.record_reference_prediction(run_id, mean_bias=-0.5, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    # left 'running', never finished -- must not count

    with pytest.raises(ValueError):
        handedness.load_b0(lg, "malecns", "real", "cfg1")
    with pytest.raises(ValueError):
        handedness.load_b0(lg, "malecns", "real", "cfg2")  # different config hash entirely
    lg.close()
