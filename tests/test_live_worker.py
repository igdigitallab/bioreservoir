"""worker.py: the glue between a (fake, duck-typed) `LIFNetwork` and `pipeline.compute_answer`.

Uses the REAL, already-cached MaleCNS harmonized graph (`bench.graph_to_arrays`, reads
`data/processed/malecns-min5/` — already on disk, ~1s, no write) and the real local encoder —
same convention as `tests/test_encode.py` (annotation/embedding loading is not a simulation). The
network itself is a `FakeNet`: deterministic, no Brian2, no full-brain simulation, matching this
worktree's "no full-brain simulations" rule while still exercising every line of `worker.py`'s own
logic (population summing, handedness lookup order, b0 caching).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bioreservoir.live import lab as lab_mod
from bioreservoir.live import states, worker
from bioreservoir.oracle import config as oracle_config
from bioreservoir.oracle.ledger import Ledger


@pytest.fixture(scope="module")
def cfg():
    return oracle_config.load_config()


@pytest.fixture(scope="module")
def full_graph(cfg):
    from bioreservoir.sim import bench

    return bench.graph_to_arrays("malecns", min_syn=cfg.trial.min_syn)


@pytest.fixture(scope="module")
def graph(full_graph):
    return full_graph[4]  # id_to_dense


@pytest.fixture(scope="module")
def readout_idx(cfg, graph):
    from bioreservoir.sim import bench

    left = bench.population_indices("malecns", graph, cfg.readout_population_left())
    right = bench.population_indices("malecns", graph, cfg.readout_population_right())
    return left, right


@pytest.fixture(scope="module")
def lab_ctx(cfg, full_graph, graph, readout_idx):
    """A real `LabContext` (annotation lookups from the actual cached MaleCNS graph, matching
    `worker.build_resources`'s own construction) — used by `answer_question` tests so `Answer.lab`
    reflects real cell types/super classes/graph metadata, not synthetic stand-ins."""
    n_neurons, pre_idx, _post_idx, weight, id_to_dense = full_graph
    left_idx, right_idx = readout_idx
    cell_type_lookup, super_class_lookup, modality_lookup = worker.build_annotation_lookups("malecns", id_to_dense)
    return lab_mod.LabContext(
        brain_name=worker.BRAIN_DISPLAY_NAME,
        n_neurons=n_neurons,
        n_connections=len(pre_idx),
        n_synapses=int(np.abs(weight).sum()),
        min_syn=cfg.trial.min_syn,
        dt_ms=0.1,
        code_sha="testsha",
        config_hash="testcfg",
        cell_type_lookup=cell_type_lookup,
        super_class_lookup=super_class_lookup,
        modality_lookup=modality_lookup,
        readout_left_idx=left_idx,
        readout_right_idx=right_idx,
        dense_to_atlas=None,
    )


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


class FakeNet:
    """Deterministic stand-in for `LIFNetwork`: every trial returns the SAME spike pattern
    regardless of `inputs`/`seed` (this test suite is about `worker.py`'s own plumbing, not about
    whether a real brain responds to text — that is `docs/MODEL.md`'s calibration, out of scope
    for an agent forbidden from running full-brain simulations)."""

    def __init__(self, n_neurons: int, left_idx: np.ndarray, right_idx: np.ndarray, left_spikes: int, right_spikes: int):
        self.n_neurons = n_neurons
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.left_spikes = left_spikes
        self.right_spikes = right_spikes
        self.calls: list[int] = []
        self.input_calls: list[tuple] = []  # (dense_idx, rate_hz) per call, F10: prove the SAME input reaches every contender

    def run_trial(self, inputs, duration_ms, seed):
        self.calls.append(seed)
        self.input_calls.append(inputs)
        counts = np.zeros(self.n_neurons, dtype=np.int64)
        if self.left_idx is not None and self.left_idx.size:
            counts[self.left_idx] = self.left_spikes
        if self.right_idx is not None and self.right_idx.size:
            counts[self.right_idx] = self.right_spikes
        return SimpleNamespace(spike_counts=counts, spike_neuron_idx=None, spike_time_ms=None)


def _row(id_, question):
    return {"id": id_, "question": question}


# -- answer_question ------------------------------------------------------------------------------


def test_answer_question_produces_a_full_answer_dict(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, left_spikes=50, right_spikes=10)
    answer = worker.answer_question(
        _row(1, "Will it rain tomorrow?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
    )
    assert answer["id"] == 1
    assert answer["answer"] in ("yes", "no")
    assert answer["n_trials"] == 3  # config.LIVE_N_TRIALS, not exp 001's trial.n_trials
    assert answer["brain"] == "malecns"
    assert net.calls  # the fake network was actually driven
    assert answer["lab"]["provenance"]["n_neurons"] == len(graph)
    # A paste-ready command against the public repository (lab.REPO_URL's shipped default); its
    # exact shape and shell-escaping are covered by the round-trip test below.
    assert answer["lab"]["provenance"]["reproduce"].startswith(
        "git clone https://github.com/igdigitallab/bioreservoir && cd bioreservoir && git checkout snapshot-testsha"
    )


def test_answer_question_runs_exactly_live_n_trials(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    from bioreservoir.live import config as live_config

    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, left_spikes=10, right_spikes=10)
    worker.answer_question(
        _row(2, "Is the sky blue?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
    )
    assert len(net.calls) == live_config.LIVE_N_TRIALS


def test_answer_question_seeds_are_deterministic_given_the_same_row_id(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net1 = FakeNet(len(graph), left_idx, right_idx, 10, 10)
    net2 = FakeNet(len(graph), left_idx, right_idx, 10, 10)
    a1 = worker.answer_question(_row(5, "Q?"), net1, graph, cfg, 0.0, lab_ctx, state_specs, state_population_idx, encoder_model)
    a2 = worker.answer_question(_row(5, "Q?"), net2, graph, cfg, 0.0, lab_ctx, state_specs, state_population_idx, encoder_model)
    assert net1.calls == net2.calls
    assert [t["seed"] for t in a1["lab"]["trials"]] == [t["seed"] for t in a2["lab"]["trials"]]
    assert a1["lab"]["raster"]["spikes_b64"] == a2["lab"]["raster"]["spikes_b64"]


def test_answer_question_left_dominant_spikes_read_as_a_leftward_bias(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    """With b0=0 and a strongly left-dominant fake readout, the corrected bias must be positive
    (the raw README.md convention: `(L-R)/(L+R)`), independent of which text/id was asked."""
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, left_spikes=100, right_spikes=1)
    answer = worker.answer_question(
        _row(3, "Will it rain tomorrow?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
    )
    # lateral_bias is the *corrected*, question-id-signed value (pipeline.py), but its magnitude
    # must be large and near the raw (100-1)/(100+1) bias regardless of the id's own coin (b0=0
    # here, so no handedness correction is even applied) -- check magnitude, not sign, since the
    # sign depends on this row id's own side-mapping coin.
    assert abs(answer["lateral_bias"]) > 0.9
    assert answer["confidence"] > 0.9


def test_answer_question_sums_state_populations_across_trials(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 10, 10)
    answer = worker.answer_question(
        _row(4, "Will it rain tomorrow?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
    )
    # arousal is the one validated state in the committed live-states.yaml -> always a number.
    assert answer["states"]["arousal"] is not None
    assert 0.0 <= answer["states"]["arousal"] <= 1.0
    for name in ("appetite", "fear", "backoff", "courtship"):
        assert answer["states"][name] is None  # unvalidated -> published null (task brief)


# -- Answer.lab ------------------------------------------------------------------------------------


def test_answer_question_lab_stimulated_matches_the_real_encoded_input(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 10, 10)
    answer = worker.answer_question(
        _row(6, "Will it rain tomorrow?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
    )
    stimulated = answer["lab"]["stimulated"]
    # config.yaml: input.n_per_side=250 -> up to 500 total (bounded by the smaller side's
    # available count, balanced_lateral_population's own rule).
    assert stimulated["total"] == stimulated["left"] + stimulated["right"]
    assert stimulated["left"] <= 250
    assert stimulated["right"] <= 250
    assert stimulated["total"] > 0
    assert sum(stimulated["by_modality"].values()) == stimulated["total"]


def test_answer_question_lab_total_spikes_and_active_fraction_are_internally_consistent(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 5, 5)
    answer = worker.answer_question(
        _row(7, "Will it rain tomorrow?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
    )
    lab = answer["lab"]
    # FakeNet drives only the readout populations -> every trial's total spike count is exactly
    # (left readout count x 5) + (right readout count x 5), summed over 3 trials.
    per_trial_total = left_idx.size * 5 + right_idx.size * 5
    assert lab["total_spikes"] == per_trial_total * 3
    assert 0.0 <= lab["active_fraction"] <= 1.0
    assert lab["active_neurons"] == round(lab["active_fraction"] * lab_ctx.n_neurons)


def test_answer_question_lab_provenance_reproduce_command_round_trips_the_id_and_question(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, monkeypatch):
    # The repository is private, so `lab.REPO_URL` ships empty and the command is None
    # (tested in test_live_lab.py). This covers the shape the panel shows once it goes public.
    monkeypatch.setattr(lab_mod, "REPO_URL", "https://github.com/igdigitallab/bioreservoir")
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 10, 10)
    question = "Will it rain tomorrow?"
    answer = worker.answer_question(
        _row(8, question), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
    )
    prov = answer["lab"]["provenance"]
    assert prov["reproduce"] == (
        f"git clone {lab_mod.REPO_URL} && cd bioreservoir && git checkout snapshot-testsha && "
        "uv sync --extra sim --extra encode --extra live && python scripts/fetch_data.py && "
        f"uv run python -m bioreservoir.live.reproduce --id 8 --question {question!r}"
    )
    assert prov["n_connections"] == lab_ctx.n_connections
    assert prov["n_synapses"] == lab_ctx.n_synapses
    assert prov["min_syn"] == cfg.trial.min_syn


# -- b0: read-only main-ledger lookup, local cache, local computation ----------------------------


def test_read_only_reference_mean_biases_reads_a_throwaway_ledger(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    lg = Ledger(path=ledger_path)
    lg.upsert_reference_sentence("ref-01", "The kettle is on the stove.")
    run_id = lg.start_reference_run("ref-01", "malecns", "real", "cfgabc", trial_seeds=[1])
    lg.record_reference_prediction(run_id, mean_bias=-0.05, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    lg.finish_reference_run(run_id, "done")
    lg.close()

    biases = worker.read_only_reference_mean_biases(ledger_path, "malecns", "real", "cfgabc")
    assert biases == pytest.approx([-0.05])


def test_read_only_reference_mean_biases_returns_empty_for_missing_file(tmp_path):
    assert worker.read_only_reference_mean_biases(tmp_path / "nope.sqlite", "malecns", "real", "x") == []


def test_read_only_reference_mean_biases_ignores_other_config_hashes(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    lg = Ledger(path=ledger_path)
    lg.upsert_reference_sentence("ref-01", "x")
    run_id = lg.start_reference_run("ref-01", "malecns", "real", "cfg1", trial_seeds=[])
    lg.record_reference_prediction(run_id, mean_bias=0.1, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    lg.finish_reference_run(run_id, "done")
    lg.close()
    assert worker.read_only_reference_mean_biases(ledger_path, "malecns", "real", "cfg-other") == []


def test_b0_cache_round_trips(tmp_path):
    path = tmp_path / "b0_cache.json"
    assert worker.load_b0_cache(path, "malecns", "real", "cfg1") is None
    worker.save_b0_cache(path, "malecns", "real", "cfg1", 0.0123)
    assert worker.load_b0_cache(path, "malecns", "real", "cfg1") == pytest.approx(0.0123)
    assert worker.load_b0_cache(path, "malecns", "real", "cfg2") is None  # different config hash


def test_compute_b0_locally_runs_every_reference_sentence(cfg, graph, readout_idx, encoder_model):
    left_idx, right_idx = readout_idx
    # 100:1 skew -> raw per-trial bias = (100*n_left - 1*n_right) / (100*n_left + 1*n_right),
    # close to 1 regardless of the (near-equal, docs/MODEL.md: ~656 vs ~648) left/right counts.
    net = FakeNet(len(graph), left_idx, right_idx, left_spikes=100, right_spikes=1)
    b0 = worker.compute_b0_locally(net, graph, cfg, left_idx, right_idx, encoder_model)
    assert isinstance(b0, float)
    # 24 reference sentences x cfg.trial.n_trials each.
    from bioreservoir.oracle import reference

    n_sentences = len(reference.load_reference(cfg.handedness.reference_path()))
    assert n_sentences == 24
    assert len(net.calls) == n_sentences * cfg.trial.n_trials
    # FakeNet is text-independent and strongly left-dominant -> b0 must be strongly positive.
    assert b0 > 0.9


def test_get_or_compute_b0_prefers_main_ledger_over_everything_else(monkeypatch, tmp_path, cfg, graph, readout_idx, encoder_model):
    from bioreservoir.live import config as live_config

    ledger_path = tmp_path / "main_ledger.sqlite"
    lg = Ledger(path=ledger_path)
    lg.upsert_reference_sentence("ref-01", "x")
    run_id = lg.start_reference_run("ref-01", "malecns", "real", "cfg-live", trial_seeds=[])
    lg.record_reference_prediction(run_id, mean_bias=-0.42, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    lg.finish_reference_run(run_id, "done")
    lg.close()
    monkeypatch.setattr(live_config, "MAIN_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(worker.config, "MAIN_LEDGER_PATH", ledger_path)

    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 100, 1)  # would give a very different b0 if used
    b0 = worker.get_or_compute_b0(net, graph, cfg, "cfg-live", left_idx, right_idx, encoder_model)
    assert b0 == pytest.approx(-0.42)
    assert net.calls == []  # never touched the fake network -- the ledger answer was authoritative


def test_get_or_compute_b0_falls_back_to_local_cache_then_computation(monkeypatch, tmp_path, cfg, graph, readout_idx, encoder_model):
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "MAIN_LEDGER_PATH", tmp_path / "nope.sqlite")
    monkeypatch.setattr(worker.config, "MAIN_LEDGER_PATH", tmp_path / "nope.sqlite")
    cache_path = tmp_path / "b0_cache.json"
    monkeypatch.setattr(live_config, "B0_CACHE_PATH", cache_path)
    monkeypatch.setattr(worker.config, "B0_CACHE_PATH", cache_path)

    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 100, 1)
    b0_first = worker.get_or_compute_b0(net, graph, cfg, "cfg-x", left_idx, right_idx, encoder_model)
    assert net.calls  # had to compute locally
    assert cache_path.exists()

    net2 = FakeNet(len(graph), left_idx, right_idx, 1, 100)  # would give the opposite sign
    b0_second = worker.get_or_compute_b0(net2, graph, cfg, "cfg-x", left_idx, right_idx, encoder_model)
    assert b0_second == pytest.approx(b0_first)
    assert net2.calls == []  # cache hit -- never touched the second fake network


def test_get_or_compute_b0_real_and_er_conditions_are_independent_cache_entries(tmp_path, monkeypatch, cfg, graph, readout_idx, encoder_model):
    from bioreservoir.live import config as live_config

    monkeypatch.setattr(live_config, "MAIN_LEDGER_PATH", tmp_path / "nope.sqlite")
    monkeypatch.setattr(worker.config, "MAIN_LEDGER_PATH", tmp_path / "nope.sqlite")
    cache_path = tmp_path / "b0_cache.json"
    monkeypatch.setattr(live_config, "B0_CACHE_PATH", cache_path)
    monkeypatch.setattr(worker.config, "B0_CACHE_PATH", cache_path)

    left_idx, right_idx = readout_idx
    real_net = FakeNet(len(graph), left_idx, right_idx, 100, 1)
    er_net = FakeNet(len(graph), left_idx, right_idx, 1, 100)  # opposite sign -- must not collide with "real"'s cache entry
    real_b0 = worker.get_or_compute_b0(real_net, graph, cfg, "cfg-y", left_idx, right_idx, encoder_model, condition="real")
    er_b0 = worker.get_or_compute_b0(er_net, graph, cfg, "cfg-y", left_idx, right_idx, encoder_model, condition="er")
    assert real_b0 > 0
    assert er_b0 < 0
    assert real_b0 != pytest.approx(er_b0)
    # Re-querying "real" must still hit its own cache entry, not the "er" one just written.
    assert worker.get_or_compute_b0(FakeNet(len(graph), left_idx, right_idx, 100, 1), graph, cfg, "cfg-y", left_idx, right_idx, encoder_model, condition="real") == pytest.approx(real_b0)


# -- no_brain: its own handedness, no LIFNetwork involved at all --------------------------------


def test_compute_no_brain_b0_locally_runs_every_reference_sentence(cfg, graph, encoder_model):
    b0 = worker.compute_no_brain_b0_locally(graph, cfg, encoder_model)
    assert isinstance(b0, float)


def test_get_or_compute_no_brain_b0_prefers_ledger_then_cache_then_computes(tmp_path, monkeypatch, cfg, graph, encoder_model):
    from bioreservoir.live import config as live_config
    from bioreservoir.oracle.ledger import Ledger

    ledger_path = tmp_path / "main_ledger.sqlite"
    lg = Ledger(path=ledger_path)
    lg.upsert_reference_sentence("ref-01", "x")
    run_id = lg.start_reference_run("ref-01", "malecns", "no_brain", "cfg-nb", trial_seeds=[])
    lg.record_reference_prediction(run_id, mean_bias=0.07, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[])
    lg.finish_reference_run(run_id, "done")
    lg.close()
    monkeypatch.setattr(live_config, "MAIN_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(worker.config, "MAIN_LEDGER_PATH", ledger_path)

    assert worker.get_or_compute_no_brain_b0(graph, cfg, "cfg-nb", encoder_model) == pytest.approx(0.07)

    # A different config hash has nothing in the ledger and no cache yet -> computed locally
    # (real reference.yaml sentences, no LIFNetwork -- cheap and deterministic given fixed data).
    monkeypatch.setattr(live_config, "MAIN_LEDGER_PATH", tmp_path / "nope.sqlite")
    monkeypatch.setattr(worker.config, "MAIN_LEDGER_PATH", tmp_path / "nope.sqlite")
    cache_path = tmp_path / "b0_cache.json"
    monkeypatch.setattr(live_config, "B0_CACHE_PATH", cache_path)
    monkeypatch.setattr(worker.config, "B0_CACHE_PATH", cache_path)
    computed = worker.get_or_compute_no_brain_b0(graph, cfg, "cfg-other", encoder_model)
    assert isinstance(computed, float)
    assert cache_path.exists()


# -- the game: answer_question with a duck-typed fake ER pool (no multiprocessing/Brian2 anywhere
# in this test file -- see test_game.py's module docstring for why the real ErContenderPool is
# not exercised here) --------------------------------------------------------------------------


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class FakeErPool:
    """Duck-typed stand-in for `game.ErContenderPool`: `submit_trials(...)` returns an
    already-resolved "future" carrying a fixed `(left, right)` pair per trial, and records every
    call (seeds AND the encoded input) so tests can assert `answer_question` submitted the SAME
    seeds AND the SAME `dense_idx`/`rate_hz` it drove the real (fake) network with (2026-09-19
    logic review F10: the old fixture only checked seeds, never the input itself)."""

    def __init__(self, left_spikes: int, right_spikes: int):
        self.left_spikes = left_spikes
        self.right_spikes = right_spikes
        self.calls: list[list[int]] = []
        self.input_calls: list[tuple] = []  # (dense_idx, rate_hz) per submit_trials call
        self.killed = False

    def submit_trials(self, dense_idx, rate_hz, duration_ms, seeds):
        self.calls.append(list(seeds))
        self.input_calls.append((dense_idx, rate_hz))
        return _ImmediateFuture([(self.left_spikes, self.right_spikes) for _ in seeds])

    def kill(self) -> None:
        # R1 (round-2 logic review): worker.run_loop's ErContenderUnavailable handler and
        # worker.build_resources' startup except branch both call `er_pool.kill()` before exiting
        # -- every duck-typed fake standing in for game.ErContenderPool needs this method too, or
        # a real ER-unavailable path raises AttributeError instead of exercising the intended fix.
        self.killed = True

    def shutdown(self) -> None:
        pass


def test_answer_question_without_er_pool_omits_the_game_field(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 10, 10)
    answer = worker.answer_question(
        _row(10, "Will it rain tomorrow?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
    )
    assert "game" not in answer


def test_answer_question_with_er_pool_adds_the_game_field(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 60, 40)
    er_pool = FakeErPool(30, 70)
    answer = worker.answer_question(
        _row(11, "Will it rain tomorrow?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
        er_pool=er_pool, random_graph_b0=0.0, no_brain_b0=0.0,
    )
    assert "game" in answer
    from bioreservoir.live import game as game_mod

    assert set(answer["game"]["contenders"].keys()) == set(game_mod.CONTENDER_NAMES)
    assert answer["game"]["contenders"]["real"]["answer"] == answer["answer"]


def test_answer_question_submits_the_er_pool_with_the_same_seeds_as_the_real_trials(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    """Task brief: "the same per-question Poisson seeds ... shared" -- the ER pool must see the
    EXACT same seed list `answer_question` drives the real (fake) network with, for this
    question."""
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 60, 40)
    er_pool = FakeErPool(30, 70)
    worker.answer_question(
        _row(12, "Is the sky blue?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
        er_pool=er_pool, random_graph_b0=0.0, no_brain_b0=0.0,
    )
    assert len(er_pool.calls) == 1
    assert er_pool.calls[0] == net.calls


def test_answer_question_er_pool_is_submitted_before_the_real_trials_run(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    """Concurrency contract: the ER submission must happen BEFORE the real trial loop runs, not
    after -- otherwise the two never overlap and the game doubles per-question wall time (task
    brief). A pool that records a shared call-order list, alongside FakeNet doing the same,
    proves the submit call precedes the first real `run_trial` call."""
    left_idx, right_idx = readout_idx
    call_order: list[str] = []

    class OrderTrackingErPool(FakeErPool):
        def submit_trials(self, dense_idx, rate_hz, duration_ms, seeds):
            call_order.append("er_submit")
            return super().submit_trials(dense_idx, rate_hz, duration_ms, seeds)

    class OrderTrackingNet(FakeNet):
        def run_trial(self, inputs, duration_ms, seed):
            call_order.append("real_trial")
            return super().run_trial(inputs, duration_ms, seed)

    net = OrderTrackingNet(len(graph), left_idx, right_idx, left_spikes=10, right_spikes=10)
    er_pool = OrderTrackingErPool(10, 10)
    worker.answer_question(
        _row(13, "Is the sky blue?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
        er_pool=er_pool, random_graph_b0=0.0, no_brain_b0=0.0,
    )
    assert call_order[0] == "er_submit"
    assert call_order[1:] == ["real_trial"] * worker.config.LIVE_N_TRIALS


def test_answer_question_er_contender_receives_the_same_encoded_input_as_the_real_brain(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    """2026-09-19 logic review F10: the old fixture proved the seeds matched but silently allowed
    a DIFFERENT encoding to reach the ER contender (it never looked at `dense_idx`/`rate_hz` at
    all). The real brain and the random-graph control must see the identical stimulus, not just
    identical trial seeds."""
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 60, 40)
    er_pool = FakeErPool(30, 70)
    worker.answer_question(
        _row(14, "Is the sky blue?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
        er_pool=er_pool, random_graph_b0=0.0, no_brain_b0=0.0,
    )
    assert len(er_pool.input_calls) == 1
    er_dense_idx, er_rate_hz = er_pool.input_calls[0]
    # Every real trial call got the SAME (dense_idx, rate_hz) too (encode_question_to_input is
    # deterministic and called once) -- compare the ER submission against the real net's own.
    for real_dense_idx, real_rate_hz in net.input_calls:
        assert np.array_equal(er_dense_idx, real_dense_idx)
        assert np.array_equal(er_rate_hz, real_rate_hz)


# -- the game: b0 missing must disable the game, never silently default to 0.0 (F7) --------------


def test_answer_question_missing_random_graph_b0_disables_the_game_and_never_touches_the_pool(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 60, 40)
    er_pool = FakeErPool(30, 70)
    answer = worker.answer_question(
        _row(15, "Is the sky blue?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
        er_pool=er_pool, random_graph_b0=None, no_brain_b0=0.0,
    )
    assert "game" not in answer
    assert er_pool.calls == []  # never submitted -- the missing b0 was caught before that


def test_answer_question_missing_no_brain_b0_disables_the_game(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 60, 40)
    er_pool = FakeErPool(30, 70)
    answer = worker.answer_question(
        _row(16, "Is the sky blue?"), net, graph, cfg, 0.0, lab_ctx,
        state_specs, state_population_idx, encoder_model,
        er_pool=er_pool, random_graph_b0=0.0, no_brain_b0=None,
    )
    assert "game" not in answer
    assert er_pool.calls == []


# -- the game: a broken/hung ER pool must raise game.ErContenderUnavailable, never hang forever
# or silently drop the question (F3/F4) -----------------------------------------------------------


class _RaisingFuture:
    def __init__(self, exc: Exception):
        self._exc = exc

    def result(self, timeout=None):
        raise self._exc


class RaisingErPool(FakeErPool):
    """Submits normally but the returned future raises on `.result()` -- simulates a hung
    (TimeoutError) or dead (BrokenProcessPool) child without any real multiprocessing."""

    def __init__(self, exc: Exception):
        super().__init__(0, 0)
        self._exc = exc

    def submit_trials(self, dense_idx, rate_hz, duration_ms, seeds):
        self.calls.append(list(seeds))
        self.input_calls.append((dense_idx, rate_hz))
        return _RaisingFuture(self._exc)


def test_answer_question_er_timeout_raises_er_contender_unavailable(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    from concurrent.futures import TimeoutError as FutureTimeoutError

    from bioreservoir.live import game as game_mod

    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 60, 40)
    er_pool = RaisingErPool(FutureTimeoutError("hung"))
    with pytest.raises(game_mod.ErContenderUnavailable):
        worker.answer_question(
            _row(17, "Is the sky blue?"), net, graph, cfg, 0.0, lab_ctx,
            state_specs, state_population_idx, encoder_model,
            er_pool=er_pool, random_graph_b0=0.0, no_brain_b0=0.0,
        )


def test_answer_question_er_broken_pool_raises_er_contender_unavailable(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    from concurrent.futures.process import BrokenProcessPool

    from bioreservoir.live import game as game_mod

    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 60, 40)
    er_pool = RaisingErPool(BrokenProcessPool("dead child"))
    with pytest.raises(game_mod.ErContenderUnavailable):
        worker.answer_question(
            _row(18, "Is the sky blue?"), net, graph, cfg, 0.0, lab_ctx,
            state_specs, state_population_idx, encoder_model,
            er_pool=er_pool, random_graph_b0=0.0, no_brain_b0=0.0,
        )


class BrokenAtSubmitErPool:
    """`submit_trials` itself raises immediately -- the pool.submit() call itself is broken
    (already-dead child), not just the returned future."""

    def submit_trials(self, dense_idx, rate_hz, duration_ms, seeds):
        from bioreservoir.live.game import ErContenderUnavailable

        raise ErContenderUnavailable("already broken")


def test_answer_question_er_broken_at_submit_propagates_before_real_trials_run(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    from bioreservoir.live import game as game_mod

    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 60, 40)
    with pytest.raises(game_mod.ErContenderUnavailable):
        worker.answer_question(
            _row(19, "Is the sky blue?"), net, graph, cfg, 0.0, lab_ctx,
            state_specs, state_population_idx, encoder_model,
            er_pool=BrokenAtSubmitErPool(), random_graph_b0=0.0, no_brain_b0=0.0,
        )
    # The real net was never driven -- a pool broken at submit time fails fast, before spending
    # any wall time on the real brain's own trials.
    assert net.calls == []


# -- run_loop: a fatal ER failure marks the question failed AND exits the process (F3/F4) --------


def test_run_loop_er_contender_unavailable_marks_failed_and_exits_nonzero(tmp_path, monkeypatch, cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    from concurrent.futures import TimeoutError as FutureTimeoutError

    from bioreservoir.live import config as live_config
    from bioreservoir.live.store import LiveStore

    monkeypatch.setattr(live_config, "WORKER_HEARTBEAT_PATH", tmp_path / "worker.heartbeat")
    monkeypatch.setattr(worker.config, "WORKER_HEARTBEAT_PATH", tmp_path / "worker.heartbeat")

    live_store = LiveStore(path=tmp_path / "live.sqlite")
    id_ = live_store.enqueue("Is the sky blue?", ip_hash="h1")

    left_idx, right_idx = readout_idx
    net = FakeNet(len(graph), left_idx, right_idx, 60, 40)
    er_pool = RaisingErPool(FutureTimeoutError("hung"))
    resources = {
        "net": net, "id_to_dense": graph, "cfg": cfg, "b0": 0.0, "lab_ctx": lab_ctx,
        "state_specs": state_specs, "state_population_idx": state_population_idx,
        "encoder_model": encoder_model, "er_pool": er_pool, "random_graph_b0": 0.0, "no_brain_b0": 0.0,
    }
    with pytest.raises(SystemExit) as exc_info:
        worker.run_loop(resources, live_store, once=True)
    assert exc_info.value.code == 1
    # R1: the fatal-exit path must kill the pool's children BEFORE raising, not just shut down --
    # this is the assertion the pre-R1 code would have failed (it never called kill() at all).
    assert er_pool.killed is True

    row = live_store.get(id_)
    assert row["status"] == "rejected"
    assert row["reason"] == "spam"  # mark_failed's own convention, see store.py
    assert "again" in row["message"].lower()
    live_store.close()


def test_run_loop_touches_the_heartbeat_file(tmp_path, monkeypatch, cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    from bioreservoir.live import config as live_config
    from bioreservoir.live.store import LiveStore

    heartbeat_path = tmp_path / "hb" / "worker.heartbeat"
    monkeypatch.setattr(live_config, "WORKER_HEARTBEAT_PATH", heartbeat_path)
    monkeypatch.setattr(worker.config, "WORKER_HEARTBEAT_PATH", heartbeat_path)

    live_store = LiveStore(path=tmp_path / "live.sqlite")
    assert not heartbeat_path.exists()
    worker.run_loop({}, live_store, once=True)  # empty queue -> touches heartbeat, then returns (once=True)
    assert heartbeat_path.exists()
    live_store.close()
