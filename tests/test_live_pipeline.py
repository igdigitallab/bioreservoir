"""pipeline.py: `compute_answer` — pure Answer-dict construction from already-run trial results,
including the `lab` object (operator requirement: "every number shown must come from the
simulation or committed docs"). No Brian2, no real connectome; reuses
`oracle.readout`/`handedness`/`side_mapping` and `live.lab` directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from bioreservoir.live import lab as lab_mod
from bioreservoir.live import pipeline, states
from bioreservoir.live.frames import decode_bin_b64
from bioreservoir.oracle import side_mapping

STATE_SPECS = states.load_states()
N_NEURONS = 1000
EMPTY_STIM = np.array([], dtype=np.int64)


def _trial(left, right, **kwargs):
    return pipeline.LiveTrial(left_spikes=left, right_spikes=right, **kwargs)


def _lab_ctx(
    n_neurons=N_NEURONS,
    readout_left_idx=None,
    readout_right_idx=None,
    dense_to_atlas=None,
    cell_type_lookup=None,
    super_class_lookup=None,
    modality_lookup=None,
):
    return lab_mod.LabContext(
        brain_name="MaleCNS v1.0",
        n_neurons=n_neurons,
        n_connections=6_235_682,
        n_synapses=89_731_551,
        min_syn=5,
        dt_ms=0.1,
        code_sha="deadbeef",
        config_hash="cfgtest",
        cell_type_lookup=cell_type_lookup or {},
        super_class_lookup=super_class_lookup or {},
        modality_lookup=modality_lookup or {},
        readout_left_idx=np.asarray(readout_left_idx if readout_left_idx is not None else [], dtype=np.int64),
        readout_right_idx=np.asarray(readout_right_idx if readout_right_idx is not None else [], dtype=np.int64),
        dense_to_atlas=dense_to_atlas,
    )


def _compute(trials, id_=1, question="Will it rain?", b0=0.0, lab_ctx=None, stim=None, stim_left=None, stim_right=None, **kwargs):
    return pipeline.compute_answer(
        id_=id_,
        question=question,
        trials=trials,
        b0=b0,
        state_specs=STATE_SPECS,
        duration_ms=250.0,
        lab_ctx=lab_ctx or _lab_ctx(),
        stimulated_dense_idx=stim if stim is not None else EMPTY_STIM,
        stimulated_left_idx=stim_left if stim_left is not None else EMPTY_STIM,
        stimulated_right_idx=stim_right if stim_right is not None else EMPTY_STIM,
        **kwargs,
    )


def test_confidence_from_bias_is_monotonic_and_bounded():
    assert pipeline.confidence_from_bias(None) == 0.5
    assert pipeline.confidence_from_bias(0.0) == pytest.approx(0.5)
    low = pipeline.confidence_from_bias(0.05)
    mid = pipeline.confidence_from_bias(0.15)
    high = pipeline.confidence_from_bias(0.30)
    assert 0.5 <= low < mid < high <= 1.0
    # Beyond the calibration scale, confidence saturates at 1.0, never exceeds it.
    assert pipeline.confidence_from_bias(10.0) == pytest.approx(1.0)
    # Sign doesn't matter, only magnitude.
    assert pipeline.confidence_from_bias(-0.2) == pipeline.confidence_from_bias(0.2)


def test_answer_yes_no_threshold():
    assert pipeline.answer_yes_no(0.51) == "yes"
    assert pipeline.answer_yes_no(0.5) == "yes"  # tie -> yes (>=)
    assert pipeline.answer_yes_no(0.49) == "no"


def test_compute_answer_schema_has_every_documented_field():
    trials = [_trial(10, 5, seed=100 + t, total_spikes=200) for t in range(3)]
    answer = _compute(trials)
    for key in ("id", "question", "answer", "confidence", "lateral_bias", "states", "frames", "brain", "sim_ms", "n_trials", "answered_at", "lab"):
        assert key in answer
    assert answer["answer"] in ("yes", "no")
    assert 0.5 <= answer["confidence"] <= 1.0
    assert answer["brain"] == "malecns"
    assert answer["n_trials"] == 3
    assert answer["sim_ms"] == 250.0
    assert answer["frames"] is None  # no dense_to_atlas given


def test_compute_answer_subtracts_handedness_before_deciding():
    # Raw bias +0.2 with a group handedness of +0.2 nets to 0 -> right at the tie boundary.
    # left_is_yes for this id/side_base decides whether that tie reads "yes" or "no"; either way
    # the corrected bias itself must be ~0, not the raw +0.2.
    trials = [_trial(60, 40) for _ in range(3)]  # raw bias = (60-40)/(100+eps) ~= 0.2
    answer = _compute(trials, id_=42, question="Q?", b0=0.2)
    assert answer["lateral_bias"] == pytest.approx(0.0, abs=1e-3)
    assert answer["confidence"] == pytest.approx(0.5, abs=1e-3)


def test_compute_answer_all_zero_spike_trials_is_a_neutral_tie():
    trials = [_trial(0, 0) for _ in range(3)]
    answer = _compute(trials, id_=7, question="Q?")
    assert answer["lateral_bias"] == 0.0
    assert answer["confidence"] == 0.5


def _yes_left(question: str) -> bool:
    return side_mapping.left_is_yes_for("q:" + pipeline.question_key(question), pipeline.config.LIVE_SIDE_BASE)


def test_compute_answer_question_text_drives_the_side_mapping_coin_deterministically():
    trials = [_trial(80, 20) for _ in range(3)]
    # Same wording under different queue ids -> same coin -> same answer (the 2026-09-18 bug:
    # the same question answered NO, then YES, when the coin was keyed by the row id).
    a1 = _compute(trials, id_=100, question="Will it rain tomorrow?")
    a2 = _compute(trials, id_=7, question="  will it RAIN tomorrow  ")
    assert a1["answer"] == a2["answer"]
    assert a1["lab"]["yes_side"] == a2["lab"]["yes_side"]

    # A strongly left-biased result reads opposite ways for two questions with opposite coins.
    questions = [f"Question number {i}?" for i in range(20)]
    yes_q = [q for q in questions if _yes_left(q)]
    no_q = [q for q in questions if not _yes_left(q)]
    assert yes_q and no_q  # sanity: both mappings occur in a small range
    assert _compute(trials, id_=1, question=yes_q[0])["answer"] != _compute(trials, id_=1, question=no_q[0])["answer"]


def test_yes_side_for_is_the_same_coin_the_answer_records():
    """The page labels YES/NO on their sides while the question is still being simulated
    (`api.py` /api/now, /api/answers/{id}, the SSE `thinking` event), so the side it publishes
    before the run must be the side the finished answer reports -- otherwise the two labels swap
    over when the verdict lands. Checked against `lab.yes_side`, not against `yes_side_for`
    itself, and across wordings that map both ways."""
    trials = [_trial(80, 20) for _ in range(3)]
    questions = ["Will it rain tomorrow?", "  will it RAIN tomorrow  "] + [f"Question number {i}?" for i in range(20)]
    sides = {pipeline.yes_side_for(q) for q in questions}
    assert sides == {"left", "right"}  # a constant answer would pass the loop below trivially
    for question in questions:
        assert pipeline.yes_side_for(question) == _compute(trials, question=question)["lab"]["yes_side"]


def test_question_key_normalizes_case_space_and_trailing_punctuation():
    assert pipeline.question_key("Will it rain tomorrow?") == pipeline.question_key("  will it  RAIN tomorrow ?! ")
    assert pipeline.question_key("Will it rain tomorrow?") != pipeline.question_key("Will it not rain tomorrow?")


def test_compute_answer_sums_state_spike_counts_across_trials():
    trials = [
        _trial(10, 5, state_spike_counts={"DNp01_giant_fiber": 3}, n_active_neurons=100),
        _trial(10, 5, state_spike_counts={"DNp01_giant_fiber": 7}, n_active_neurons=200),
    ]
    answer = _compute(trials, id_=1, question="Q?")
    # fear is unvalidated in the committed live-states.yaml -> still null regardless of counts.
    assert answer["states"]["fear"] is None
    # arousal IS validated -> averaged active-neuron fraction across the two trials.
    assert answer["states"]["arousal"] == pytest.approx(((100 + 200) / 2) / N_NEURONS)


def test_compute_answer_uses_frames_from_first_trial_with_recorded_spikes():
    dense_to_atlas = {i: i * 10 for i in range(10)}
    trials = [
        _trial(10, 5, spike_neuron_idx=None, spike_time_ms=None),
        _trial(10, 5, spike_neuron_idx=np.array([1, 2]), spike_time_ms=np.array([1.0, 2.0])),
        _trial(10, 5, spike_neuron_idx=np.array([9]), spike_time_ms=np.array([1.0])),
    ]
    answer = _compute(trials, id_=1, question="Q?", lab_ctx=_lab_ctx(dense_to_atlas=dense_to_atlas))
    assert answer["frames"] is not None
    bin0 = decode_bin_b64(answer["frames"]["active_b64"][0])
    # Must come from trial index 1 (dense 1,2 -> atlas 10,20), NOT trial index 2 (dense 9 -> 90).
    assert sorted(bin0.tolist()) == [10, 20]


# -- lab object ------------------------------------------------------------------------------------


def test_lab_trials_carry_seed_and_per_trial_bias():
    trials = [
        _trial(80, 20, seed=111, total_spikes=500),
        _trial(20, 80, seed=222, total_spikes=600),
        _trial(0, 0, seed=333, total_spikes=0),  # zero-spike trial -> bias reported as 0.0, not None
    ]
    answer = _compute(trials, id_=1, question="Q?")
    lab = answer["lab"]
    assert [t["seed"] for t in lab["trials"]] == [111, 222, 333]
    assert lab["trials"][0]["spikes_left"] == 80
    assert lab["trials"][0]["spikes_right"] == 20
    assert lab["trials"][0]["bias"] == pytest.approx((80 - 20) / 100, abs=1e-3)
    assert lab["trials"][2]["bias"] == 0.0  # zero-spike, not None (JSON-safe float)
    assert lab["total_spikes"] == 500 + 600 + 0


def test_lab_b0_corrected_bias_and_yes_side_match_top_level_answer():
    trials = [_trial(80, 20, seed=1) for _ in range(3)]
    answer = _compute(trials, id_=5, question="Q?", b0=0.1)
    lab = answer["lab"]
    assert lab["b0"] == pytest.approx(0.1)
    assert lab["corrected_bias"] == pytest.approx(answer["lateral_bias"])
    assert lab["yes_side"] in ("left", "right")
    expected_side = "left" if _yes_left("Q?") else "right"
    assert lab["yes_side"] == expected_side


def test_lab_stimulated_tallies_modality_and_side():
    modality_lookup = {0: "gustatory", 1: "gustatory", 2: "olfactory"}
    ctx = _lab_ctx(modality_lookup=modality_lookup)
    trials = [_trial(1, 1) for _ in range(3)]
    answer = _compute(
        trials, id_=1, question="Q?", lab_ctx=ctx,
        stim=np.array([0, 1, 2]), stim_left=np.array([0, 1]), stim_right=np.array([2]),
    )
    stimulated = answer["lab"]["stimulated"]
    assert stimulated["total"] == 3
    assert stimulated["left"] == 2
    assert stimulated["right"] == 1
    assert stimulated["by_modality"] == {"gustatory": 2, "olfactory": 1}


def test_lab_active_neurons_and_fraction_match_arousal_state_input():
    trials = [_trial(1, 1, n_active_neurons=100), _trial(1, 1, n_active_neurons=300)]
    answer = _compute(trials, id_=1, question="Q?")
    lab = answer["lab"]
    assert lab["active_neurons"] == round((100 + 300) / 2)
    assert lab["active_fraction"] == pytest.approx(((100 + 300) / 2) / N_NEURONS)


def test_lab_readout_latency_ms_is_the_mean_first_readout_spike_across_trials():
    ctx = _lab_ctx(readout_left_idx=[2], readout_right_idx=[7])
    trials = [
        _trial(1, 1, spike_neuron_idx=np.array([2, 9]), spike_time_ms=np.array([12.5, 1.0])),
        _trial(1, 1, spike_neuron_idx=np.array([7, 2]), spike_time_ms=np.array([5.0, 20.0])),
        _trial(1, 1, spike_neuron_idx=None, spike_time_ms=None),  # not recorded -> excluded
    ]
    answer = _compute(trials, id_=1, question="Q?", lab_ctx=ctx)
    # trial0 earliest readout spike (idx 2) = 12.5ms; trial1 earliest readout spike (idx 7) = 5.0ms
    assert answer["lab"]["readout_latency_ms"] == pytest.approx((12.5 + 5.0) / 2)


def test_lab_readout_latency_ms_is_none_when_readout_never_fires():
    ctx = _lab_ctx(readout_left_idx=[2], readout_right_idx=[7])
    trials = [_trial(1, 1, spike_neuron_idx=np.array([9]), spike_time_ms=np.array([1.0]))]
    answer = _compute(trials, id_=1, question="Q?", lab_ctx=ctx)
    assert answer["lab"]["readout_latency_ms"] is None


def test_lab_top_cell_types_excludes_stimulated_neurons_and_sorts_by_spikes():
    cell_type_lookup = {0: "GRN", 1: "GRN", 2: "MN9", 3: "MN9", 4: "DNp01"}
    ctx = _lab_ctx(cell_type_lookup=cell_type_lookup, super_class_lookup={2: "motor", 3: "motor", 4: "descending"})
    trials = [
        _trial(
            1, 1,
            spike_neuron_idx=np.array([0, 2, 2, 2, 3, 4]),  # 0 is stimulated -> excluded
            spike_time_ms=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
    ]
    answer = _compute(trials, id_=1, question="Q?", lab_ctx=ctx, stim=np.array([0]))
    top = answer["lab"]["top_cell_types"]
    types = [row["cell_type"] for row in top]
    assert "GRN" not in types or all(  # neuron 0 (GRN) is stimulated; neuron 1 (also GRN) never fired
        row["cell_type"] != "GRN" for row in top
    )
    mn9_row = next(row for row in top if row["cell_type"] == "MN9")
    assert mn9_row["n_neurons"] == 2  # dense idx 2 (3 spikes) and idx 3 (1 spike) both fired
    assert mn9_row["spikes"] == 4
    assert mn9_row["super_class"] == "motor"
    assert mn9_row["rate_hz"] == pytest.approx(4 / (2 * 1 * 0.25))  # 1 trial x 250ms


def test_lab_raster_selects_all_readout_neurons_plus_active_sample_capped_at_300():
    readout = list(range(5))
    ctx = _lab_ctx(readout_left_idx=readout[:2], readout_right_idx=readout[2:], dense_to_atlas={i: i + 1000 for i in readout})
    trials = [_trial(1, 1, spike_neuron_idx=np.array(readout + [50, 51]), spike_time_ms=np.array([1.0] * 7))]
    answer = _compute(trials, id_=1, question="Q?", lab_ctx=ctx)
    raster = answer["lab"]["raster"]
    assert len(raster["atlas_indices"]) <= lab_mod.RASTER_MAX_NEURONS
    assert len(raster["atlas_indices"]) == len(raster["cell_types"])
    # every readout neuron's atlas index must be present (mapped via dense_to_atlas).
    for i in readout:
        assert (i + 1000) in raster["atlas_indices"]


def test_lab_raster_spikes_b64_decodes_to_row_time_pairs():
    ctx = _lab_ctx(readout_left_idx=[0], readout_right_idx=[1])
    trials = [_trial(1, 1, spike_neuron_idx=np.array([0, 1]), spike_time_ms=np.array([0.0, 12.3]))]
    answer = _compute(trials, id_=1, question="Q?", lab_ctx=ctx)
    pairs = lab_mod.decode_raster_spikes_b64(answer["lab"]["raster"]["spikes_b64"])
    assert pairs.shape == (2, 2)
    times_ticks = sorted(pairs[:, 1].tolist())
    assert times_ticks == [0, 123]  # 12.3ms / 0.1ms = 123 ticks


def test_lab_provenance_has_every_documented_field_and_a_working_reproduce_command(monkeypatch):
    # The repository is private, so `lab.REPO_URL` ships empty and the command is None
    # (tested in test_live_lab.py). This covers the shape the panel shows once it goes public.
    monkeypatch.setattr(lab_mod, "REPO_URL", "https://github.com/igdigitallab/bioreservoir")
    ctx = _lab_ctx()
    trials = [_trial(1, 1, seed=1)]
    answer = _compute(trials, id_=99, question="Will it rain?", lab_ctx=ctx)
    prov = answer["lab"]["provenance"]
    for key in ("brain", "n_neurons", "n_connections", "n_synapses", "min_syn", "model", "dt_ms", "sim_ms", "code_sha", "config_hash", "reproduce"):
        assert key in prov
    assert prov["brain"] == "MaleCNS v1.0"
    assert prov["model"] == "LIF, Shiu et al. 2024"
    assert prov["dt_ms"] == 0.1
    assert prov["n_neurons"] == N_NEURONS
    assert "--id 99" in prov["reproduce"]
    assert "python -m bioreservoir.live.reproduce" in prov["reproduce"]
    assert "Will it rain?" in prov["reproduce"]


# -- game: Answer.game (the "which one is the real fly?" feature) ---------------------------------


def _game_inputs(**overrides):
    from bioreservoir.live import game

    base = {
        "random_graph_trials": [(60, 40), (55, 45), (58, 42)],
        "random_graph_b0": 0.0,
        "no_brain_bias": 0.05,
        "no_brain_b0": 0.0,
    }
    base.update(overrides)
    return game.GameInputs(**base)


def test_compute_answer_without_game_inputs_omits_the_game_field_entirely():
    """Backward compatibility (task brief): every caller before the game feature, and every
    answer already stored on disk, has no `game_inputs` -- the key must be ABSENT, not `null`, so
    old-shape callers/frontends never see a field they don't expect."""
    trials = [_trial(10, 5) for _ in range(3)]
    answer = _compute(trials, id_=1, question="Will it rain?")
    assert "game" not in answer


def test_compute_answer_with_game_inputs_adds_the_documented_schema():
    trials = [_trial(10, 5) for _ in range(3)]
    answer = _compute(trials, id_=1, question="Will it rain?", game_inputs=_game_inputs())
    from bioreservoir.live import game as game_mod

    assert "game" in answer
    game = answer["game"]
    assert sorted(game["order"]) == sorted(game_mod.CONTENDER_NAMES)
    assert set(game["contenders"].keys()) == set(game_mod.CONTENDER_NAMES)


def test_compute_answer_games_real_contender_matches_the_top_level_answer_exactly():
    """The real brain's own game entry must be the SAME numbers as the top-level fields (task
    brief: "keep the existing top-level fields as the real brain's answer"), not independently
    recomputed."""
    trials = [_trial(80, 20) for _ in range(3)]
    answer = _compute(trials, id_=1, question="Will it rain?", b0=0.05, game_inputs=_game_inputs())
    real = answer["game"]["contenders"]["real"]
    assert real["answer"] == answer["answer"]
    assert real["corrected_bias"] == pytest.approx(answer["lateral_bias"])
    assert real["decisiveness"] == pytest.approx(answer["confidence"])


def test_compute_answer_game_shares_the_same_yes_side_coin_as_the_real_answer():
    """Task brief: "the yes-side coin is per question, so it is shared" -- a no_brain reading with
    the SAME raw bias/b0 as the real brain's own corrected bias must land on the SAME answer,
    because both are mapped through the identical `left_is_yes` coin for this question."""
    trials = [_trial(80, 20) for _ in range(3)]  # raw bias ~= 0.6
    answer = _compute(
        trials, id_=1, question="Will it rain?", b0=0.0,
        game_inputs=_game_inputs(no_brain_bias=0.6, no_brain_b0=0.0),
    )
    assert answer["game"]["contenders"]["no_brain"]["answer"] == answer["answer"]


def test_compute_answer_game_order_is_deterministic_for_the_same_question_text():
    from bioreservoir.live import game as game_mod

    trials = [_trial(10, 5) for _ in range(3)]
    a1 = _compute(trials, id_=1, question="Will it rain tomorrow?", game_inputs=_game_inputs())
    a2 = _compute(trials, id_=2, question="  will it  RAIN tomorrow ?! ", game_inputs=_game_inputs())
    assert a1["game"]["order"] == a2["game"]["order"]
    assert a1["game"]["order"] == game_mod.contender_order("q:" + pipeline.question_key("Will it rain tomorrow?"))
