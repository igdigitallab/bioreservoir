"""reproduce.py: `python -m bioreservoir.live.reproduce` (task brief: "a CLI that recomputes one
answer from its seed/config" + "a test that it is deterministic").

`reproduce_answer` is tested directly against `worker.answer_question` (which it wraps) using the
SAME fake-network convention as `test_live_worker.py`: a duck-typed `FakeNet` stands in for
`LIFNetwork` (no Brian2 network is ever built here, no full-brain simulation), driven by the real,
already-cached MaleCNS annotation/graph-index data (population/encode lookups need real
`populations.yaml` rules — annotation loading is not a simulation, same convention `test_encode.py`
already established). `main()`'s CLI wiring itself (argument parsing -> `worker.build_resources()`)
is exercised structurally via `reproduce_answer`'s signature match to `worker.build_resources()`'s
return shape, not by actually invoking `build_resources()` (which builds a real `LIFNetwork` and
must only run in the cage, per this worktree's rules).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bioreservoir.live import lab as lab_mod
from bioreservoir.live import reproduce, states, worker
from bioreservoir.oracle import config as oracle_config


@pytest.fixture(scope="module")
def cfg():
    return oracle_config.load_config()


@pytest.fixture(scope="module")
def full_graph(cfg):
    from bioreservoir.sim import bench

    return bench.graph_to_arrays("malecns", min_syn=cfg.trial.min_syn)


@pytest.fixture(scope="module")
def graph(full_graph):
    return full_graph[4]


@pytest.fixture(scope="module")
def readout_idx(cfg, graph):
    from bioreservoir.sim import bench

    left = bench.population_indices("malecns", graph, cfg.readout_population_left())
    right = bench.population_indices("malecns", graph, cfg.readout_population_right())
    return left, right


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


@pytest.fixture(scope="module")
def lab_ctx(cfg, full_graph, graph, readout_idx):
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


class FakeNet:
    """Same deterministic FakeNet convention as `test_live_worker.py`: fixed spike pattern on
    the readout population regardless of `inputs`/`seed` -- this test suite is about
    `reproduce.py`'s own determinism (same id/question -> same seeds -> same Answer), not about
    brain physics."""

    def __init__(self, n_neurons, left_idx, right_idx, left_spikes=40, right_spikes=15):
        self.n_neurons = n_neurons
        self.left_idx = left_idx
        self.right_idx = right_idx
        self.left_spikes = left_spikes
        self.right_spikes = right_spikes
        self.calls: list[int] = []

    def run_trial(self, inputs, duration_ms, seed):
        self.calls.append(seed)
        counts = np.zeros(self.n_neurons, dtype=np.int64)
        counts[self.left_idx] = self.left_spikes
        counts[self.right_idx] = self.right_spikes
        return SimpleNamespace(spike_counts=counts, spike_neuron_idx=None, spike_time_ms=None)


def _resources(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, net):
    return {
        "net": net,
        "id_to_dense": graph,
        "cfg": cfg,
        "b0": 0.02,
        "lab_ctx": lab_ctx,
        "state_specs": state_specs,
        "state_population_idx": state_population_idx,
        "encoder_model": encoder_model,
    }


def test_reproduce_answer_is_byte_for_byte_deterministic(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    net1 = FakeNet(len(graph), left_idx, right_idx)
    net2 = FakeNet(len(graph), left_idx, right_idx)
    resources1 = _resources(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, net1)
    resources2 = _resources(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, net2)

    answer1 = reproduce.reproduce_answer(42, "Will it rain tomorrow?", resources1)
    answer2 = reproduce.reproduce_answer(42, "Will it rain tomorrow?", resources2)

    # The two FakeNet instances saw identical seeds (determinism at the seed-derivation layer)...
    assert net1.calls == net2.calls
    # ...and produced byte-for-byte identical Answer dicts (determinism end to end, including the
    # `lab` object -- raster encoding, provenance, everything) -- except `answered_at`, a real
    # wall-clock timestamp of when THIS reproduction ran, which is correctly NOT part of the
    # reproducibility contract (the simulation result is deterministic; the clock is not).
    answer1.pop("answered_at")
    answer2.pop("answered_at")
    assert answer1 == answer2


def test_reproduce_answer_matches_a_direct_worker_answer_question_call(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    """`reproduce_answer` is a thin wrapper -- confirms it is not silently doing anything
    different from `worker.answer_question` itself (e.g. missing an argument)."""
    left_idx, right_idx = readout_idx
    net1 = FakeNet(len(graph), left_idx, right_idx)
    net2 = FakeNet(len(graph), left_idx, right_idx)
    resources = _resources(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, net1)

    via_reproduce = reproduce.reproduce_answer(7, "Is the sky blue?", resources)
    via_worker = worker.answer_question(
        {"id": 7, "question": "Is the sky blue?"}, net2, graph, cfg, 0.02, lab_ctx,
        state_specs, state_population_idx, encoder_model,
    )
    via_reproduce.pop("answered_at")
    via_worker.pop("answered_at")
    assert via_reproduce == via_worker


def test_reproduce_answer_is_keyed_by_question_text_not_by_queue_id(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    # The same wording under two queue ids replays the same seeds (and so the same answer); a
    # different wording does not. Seeds used to be keyed by the row id, so one question could
    # answer NO and later YES (2026-09-18).
    left_idx, right_idx = readout_idx

    def run(id_, question):
        resources = _resources(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, FakeNet(len(graph), left_idx, right_idx))
        return reproduce.reproduce_answer(id_, question, resources)

    a, b, c = run(1, "Will it rain?"), run(2, "will it  rain"), run(3, "Will it snow?")
    seeds = lambda ans: [t["seed"] for t in ans["lab"]["trials"]]
    assert seeds(a) == seeds(b)
    assert a["answer"] == b["answer"] and a["lab"]["yes_side"] == b["lab"]["yes_side"]
    assert seeds(a) != seeds(c)


def test_provenance_reproduce_command_is_the_actual_cli_invocation(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, monkeypatch):
    # The repository is private, so `lab.REPO_URL` ships empty and the command is None
    # (tested in test_live_lab.py). This covers the shape the panel shows once it goes public.
    monkeypatch.setattr(lab_mod, "REPO_URL", "https://github.com/igdigitallab/bioreservoir")
    left_idx, right_idx = readout_idx
    resources = _resources(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, FakeNet(len(graph), left_idx, right_idx))
    answer = reproduce.reproduce_answer(9, "Will it rain tomorrow?", resources)
    command = answer["lab"]["provenance"]["reproduce"]
    assert command == (
        f"git clone {lab_mod.REPO_URL} && cd bioreservoir && git checkout snapshot-testsha && "
        "uv sync --extra sim --extra encode --extra live && python scripts/fetch_data.py && "
        "uv run python -m bioreservoir.live.reproduce --id 9 --question 'Will it rain tomorrow?'"
    )

    # The command's own arguments, if parsed and re-run, would target the same id/question this
    # test just used -- confirms the string is not decorative.
    import shlex

    parser_args = shlex.split(command)
    id_index = parser_args.index("--id")
    assert parser_args[id_index + 1] == "9"
    question_index = parser_args.index("--question")
    assert parser_args[question_index + 1] == "Will it rain tomorrow?"


# -- game: resources.get(...) degrades gracefully when the game's extra keys are absent (task
# brief: "update reproduce.py so the reproduce command recomputes all three contenders") ---------


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class FakeErPool:
    def __init__(self, left_spikes: int, right_spikes: int):
        self.left_spikes = left_spikes
        self.right_spikes = right_spikes
        self.calls: list[list[int]] = []

    def submit_trials(self, dense_idx, rate_hz, duration_ms, seeds):
        self.calls.append(list(seeds))
        return _ImmediateFuture([(self.left_spikes, self.right_spikes) for _ in seeds])


def test_reproduce_answer_without_game_resources_omits_the_game_field(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    """`resources` shaped like `worker.build_resources()`'s output BEFORE the game feature (no
    `er_pool`/`random_graph_b0`/`no_brain_b0` keys) must still reproduce the real brain's own
    answer unchanged, with `Answer.game` simply absent — not an error."""
    left_idx, right_idx = readout_idx
    resources = _resources(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, FakeNet(len(graph), left_idx, right_idx))
    answer = reproduce.reproduce_answer(20, "Will it rain tomorrow?", resources)
    assert "game" not in answer


def test_reproduce_answer_recomputes_all_three_contenders_when_game_resources_are_present(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model):
    left_idx, right_idx = readout_idx
    resources = _resources(cfg, graph, readout_idx, lab_ctx, state_specs, state_population_idx, encoder_model, FakeNet(len(graph), left_idx, right_idx))
    resources["er_pool"] = FakeErPool(30, 70)
    resources["random_graph_b0"] = 0.0
    resources["no_brain_b0"] = 0.0
    answer = reproduce.reproduce_answer(21, "Will it rain tomorrow?", resources)
    assert "game" in answer
    from bioreservoir.live import game as game_mod

    assert set(answer["game"]["contenders"].keys()) == set(game_mod.CONTENDER_NAMES)
