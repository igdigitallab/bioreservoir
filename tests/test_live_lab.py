"""lab.py: pure "lab" statistics — raster selection/encoding, top cell types, readout latency,
stimulated-population tally, provenance. No Brian2, all synthetic dense-index arrays.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bioreservoir.live import lab


def _trial(spike_neuron_idx=None, spike_time_ms=None):
    return SimpleNamespace(
        spike_neuron_idx=None if spike_neuron_idx is None else np.asarray(spike_neuron_idx, dtype=np.int64),
        spike_time_ms=None if spike_time_ms is None else np.asarray(spike_time_ms, dtype=np.float64),
    )


# -- select_raster_neurons --------------------------------------------------------------------------


def test_select_raster_neurons_includes_every_readout_neuron_when_under_cap():
    readout = np.arange(10)
    trial0 = _trial(spike_neuron_idx=[10, 11, 12], spike_time_ms=[1.0, 2.0, 3.0])
    selected = lab.select_raster_neurons(trial0, readout, seed=0, max_neurons=20)
    assert set(readout.tolist()) <= set(selected.tolist())
    assert selected.size <= 20


def test_select_raster_neurons_fills_remaining_budget_with_active_neurons():
    readout = np.arange(5)
    trial0 = _trial(spike_neuron_idx=list(range(5, 15)), spike_time_ms=[1.0] * 10)
    selected = lab.select_raster_neurons(trial0, readout, seed=0, max_neurons=8)
    assert selected.size == 8
    assert set(readout.tolist()) <= set(selected.tolist())


def test_select_raster_neurons_subsamples_readout_itself_when_over_cap():
    readout = np.arange(500)
    selected = lab.select_raster_neurons(None, readout, seed=0, max_neurons=300)
    assert selected.size == 300
    assert set(selected.tolist()) <= set(readout.tolist())


def test_select_raster_neurons_is_deterministic_given_the_same_seed():
    readout = np.arange(5)
    trial0 = _trial(spike_neuron_idx=list(range(5, 50)), spike_time_ms=[1.0] * 45)
    a = lab.select_raster_neurons(trial0, readout, seed=42, max_neurons=10)
    b = lab.select_raster_neurons(trial0, readout, seed=42, max_neurons=10)
    assert np.array_equal(a, b)


def test_select_raster_neurons_handles_no_trial0_and_no_active_neurons():
    readout = np.array([1, 2])
    selected = lab.select_raster_neurons(None, readout, seed=0, max_neurons=300)
    assert set(selected.tolist()) == {1, 2}


# -- build_raster / decode_raster_spikes_b64 --------------------------------------------------------


def test_build_raster_atlas_indices_are_minus_one_without_an_atlas():
    readout = np.array([0, 1])
    trial0 = _trial(spike_neuron_idx=[0, 1], spike_time_ms=[1.0, 2.0])
    raster = lab.build_raster(trial0, readout, dense_to_atlas=None, cell_type_lookup={}, seed=0)
    assert raster["atlas_indices"] == [-1, -1]


def test_build_raster_atlas_indices_map_present_neurons_and_flag_unmapped_as_minus_one():
    readout = np.array([0, 1, 2])
    dense_to_atlas = {0: 100, 1: 200}  # 2 is not in the atlas
    trial0 = _trial()
    raster = lab.build_raster(trial0, readout, dense_to_atlas, cell_type_lookup={}, seed=0)
    by_dense = dict(zip(sorted(readout.tolist()), raster["atlas_indices"], strict=True))
    assert by_dense[0] == 100
    assert by_dense[1] == 200
    assert by_dense[2] == -1


def test_build_raster_cell_types_default_to_unknown():
    readout = np.array([5])
    raster = lab.build_raster(_trial(), readout, None, cell_type_lookup={5: "MN9"}, seed=0)
    assert raster["cell_types"] == ["MN9"]
    raster2 = lab.build_raster(_trial(), readout, None, cell_type_lookup={}, seed=0)
    assert raster2["cell_types"] == ["unknown"]


def test_build_raster_spikes_include_every_selected_neuron_quantized_to_0_1ms():
    readout = np.array([0, 1])
    # Plenty of budget under the 300 cap, so neuron 99 (active in trial0, not readout) is also
    # selected -- "all readout neurons + seeded sample of active ones" (task brief), not "only
    # readout neurons".
    trial0 = _trial(spike_neuron_idx=[0, 1, 99], spike_time_ms=[0.0, 25.05, 5.0])
    raster = lab.build_raster(trial0, readout, None, cell_type_lookup={}, seed=0)
    assert set(raster["atlas_indices"]) == {-1}  # no atlas -> every entry is the -1 sentinel
    pairs = lab.decode_raster_spikes_b64(raster["spikes_b64"])
    assert pairs.shape == (3, 2)  # neurons 0, 1 and 99 all selected and all spiked
    times = sorted(pairs[:, 1].tolist())
    assert times == [0, 50, 250]  # 0.0ms->0, 5.0ms->50, 25.05ms->round(250.5)=250 (Python banker's rounding)


def test_build_raster_clips_time_to_uint16_range():
    readout = np.array([0])
    trial0 = _trial(spike_neuron_idx=[0], spike_time_ms=[999999.0])
    raster = lab.build_raster(trial0, readout, None, cell_type_lookup={}, seed=0)
    pairs = lab.decode_raster_spikes_b64(raster["spikes_b64"])
    assert pairs[0, 1] == 65535


def test_build_raster_empty_spikes_encodes_to_a_valid_empty_array():
    raster = lab.build_raster(_trial(), np.array([0]), None, cell_type_lookup={}, seed=0)
    pairs = lab.decode_raster_spikes_b64(raster["spikes_b64"])
    assert pairs.shape == (0, 2)


# -- top_cell_types ----------------------------------------------------------------------------------


def test_top_cell_types_excludes_stimulated_and_sorts_descending():
    trials = [_trial(spike_neuron_idx=[0, 1, 1, 2, 2, 2], spike_time_ms=[1, 2, 3, 4, 5, 6])]
    cell_type_lookup = {0: "A", 1: "B", 2: "C"}
    super_class_lookup = {1: "sc_b", 2: "sc_c"}
    rows = lab.top_cell_types(trials, stimulated_dense_idx=np.array([0]), cell_type_lookup=cell_type_lookup, super_class_lookup=super_class_lookup, n_trials=1, duration_ms=1000.0)
    assert [r["cell_type"] for r in rows] == ["C", "B"]  # C has 3 spikes, B has 2, A excluded
    assert rows[0]["spikes"] == 3
    assert rows[0]["n_neurons"] == 1
    assert rows[0]["rate_hz"] == pytest.approx(3 / (1 * 1.0))  # 1 trial x 1000ms = 1s


def test_top_cell_types_caps_at_top_n():
    trials = [_trial(spike_neuron_idx=list(range(20)), spike_time_ms=[1.0] * 20)]
    cell_type_lookup = {i: f"type{i}" for i in range(20)}
    rows = lab.top_cell_types(trials, np.array([], dtype=np.int64), cell_type_lookup, {}, n_trials=1, duration_ms=1000.0, top_n=5)
    assert len(rows) == 5


def test_top_cell_types_handles_unrecorded_trials():
    trials = [_trial()]  # spike_neuron_idx is None
    rows = lab.top_cell_types(trials, np.array([], dtype=np.int64), {}, {}, n_trials=1, duration_ms=250.0)
    assert rows == []


# -- readout_latency_ms ------------------------------------------------------------------------------


def test_readout_latency_ms_averages_first_spike_across_trials_that_have_one():
    readout = np.array([2, 7])
    trials = [
        _trial(spike_neuron_idx=[2, 9], spike_time_ms=[12.5, 1.0]),
        _trial(spike_neuron_idx=[7, 2], spike_time_ms=[5.0, 20.0]),
        _trial(),  # not recorded
    ]
    assert lab.readout_latency_ms(trials, readout) == pytest.approx((12.5 + 5.0) / 2)


def test_readout_latency_ms_none_when_no_readout_spikes():
    readout = np.array([2, 7])
    trials = [_trial(spike_neuron_idx=[9], spike_time_ms=[1.0])]
    assert lab.readout_latency_ms(trials, readout) is None


def test_readout_latency_ms_none_for_empty_readout_population():
    trials = [_trial(spike_neuron_idx=[1], spike_time_ms=[1.0])]
    assert lab.readout_latency_ms(trials, np.array([], dtype=np.int64)) is None


# -- stimulated_summary -------------------------------------------------------------------------------


def test_stimulated_summary_tallies_by_modality():
    stim = np.array([0, 1, 2, 3])
    modality_lookup = {0: "gustatory", 1: "gustatory", 2: "olfactory"}  # 3 -> unknown
    result = lab.stimulated_summary(stim, np.array([0, 1]), np.array([2, 3]), modality_lookup)
    assert result == {"total": 4, "left": 2, "right": 2, "by_modality": {"gustatory": 2, "olfactory": 1, "unknown": 1}}


# -- build_provenance --------------------------------------------------------------------------------


def test_default_repo_url_is_the_public_repository():
    """The shipped default is the public repo, so a clone of this code prints a clone command that
    actually resolves — no deploy-time environment variable needed to make it true."""
    assert lab.REPO_URL == "https://github.com/igdigitallab/bioreservoir"


def test_no_reproduce_command_when_no_repository_url_is_configured(monkeypatch):
    """A fork with no public mirror sets BIORESERVOIR_REPO_URL empty and gets no command at all,
    rather than a paste-ready `git clone` whose first step fails. The panel still renders the
    pinning hashes and says so (site/src/pages/labReadout.ts)."""
    monkeypatch.setattr(lab, "REPO_URL", "")
    ctx = lab.LabContext(
        brain_name="MaleCNS v1.0", n_neurons=100, n_connections=200, n_synapses=300, min_syn=5,
        dt_ms=0.1, code_sha="abc123", config_hash="cfgxyz",
        cell_type_lookup={}, super_class_lookup={}, modality_lookup={},
        readout_left_idx=np.array([]), readout_right_idx=np.array([]), dense_to_atlas=None,
    )
    prov = lab.build_provenance(ctx, id_=7, question="Will it rain?", sim_ms=250.0)
    assert prov["reproduce"] is None
    assert prov["code_sha"] == "abc123"  # the hashes still pin the answer
    assert prov["config_hash"] == "cfgxyz"


def test_build_provenance_reproduce_command_shell_escapes_the_question(monkeypatch):
    monkeypatch.setattr(lab, "REPO_URL", "https://github.com/igdigitallab/bioreservoir")
    ctx = lab.LabContext(
        brain_name="MaleCNS v1.0", n_neurons=100, n_connections=200, n_synapses=300, min_syn=5,
        dt_ms=0.1, code_sha="abc123", config_hash="cfgxyz",
        cell_type_lookup={}, super_class_lookup={}, modality_lookup={},
        readout_left_idx=np.array([]), readout_right_idx=np.array([]), dense_to_atlas=None,
    )
    prov = lab.build_provenance(ctx, id_=7, question="Won't this break? \"quotes\" too", sim_ms=250.0)
    # Self-contained: clone the public repo, pin the exact commit, then the actual reproduce
    # invocation (task brief: "a command that actually works against the public GitHub repo
    # layout", not just `python -m ...` assuming an already-set-up checkout).
    assert prov["reproduce"].startswith(
        f"git clone {lab.REPO_URL} && cd bioreservoir && git checkout snapshot-abc123 && "
    )
    assert "uv run python -m bioreservoir.live.reproduce --id 7 --question " in prov["reproduce"]
    assert prov["code_sha"] == "abc123"
    assert prov["config_hash"] == "cfgxyz"
    assert prov["model"] == lab.MODEL_CITATION

    import shlex

    tokens = shlex.split(prov["reproduce"])
    assert tokens[-1] == "Won't this break? \"quotes\" too"  # round-trips through the shell parser


def test_build_provenance_falls_back_to_head_when_code_sha_is_missing(monkeypatch):
    monkeypatch.setattr(lab, "REPO_URL", "https://github.com/igdigitallab/bioreservoir")
    ctx = lab.LabContext(
        brain_name="MaleCNS v1.0", n_neurons=100, n_connections=200, n_synapses=300, min_syn=5,
        dt_ms=0.1, code_sha=None, config_hash="cfgxyz",
        cell_type_lookup={}, super_class_lookup={}, modality_lookup={},
        readout_left_idx=np.array([]), readout_right_idx=np.array([]), dense_to_atlas=None,
    )
    prov = lab.build_provenance(ctx, id_=1, question="Will it rain?", sim_ms=100.0)
    assert prov["code_sha"] is None
    assert "git checkout HEAD" in prov["reproduce"]
