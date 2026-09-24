"""game.py: the "which one is the real fly?" contender-order shuffle and the pure `build_game`
assembly, PLUS (2026-09-19 logic review, F10: "ErContenderPool ... have zero tests") a real,
tiny-synthetic-graph, `codegen_target="numpy"` (no compile, fast -- same convention
`test_lif_record_spike_times.py` already established for "no full-brain simulations") exercise of
`ErContenderPool` itself: pickling across the process boundary, the readout reduction, warm-up,
and a genuine `BrokenProcessPool` (a bad initializer, not a mock) converted to
`ErContenderUnavailable`. `worker.answer_question`'s own tests (`test_live_worker.py`) separately
exercise the INTEGRATION contract (submit before the real trials, shared seeds AND input, timeout/
broken-pool handling) against a duck-typed fake pool, matching this codebase's established FakeNet
convention -- the two test files are complementary, not redundant.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from bioreservoir.live import game

# Only the ErContenderPool tests near the bottom of this file need brian2 (the `sim` extra) -- the
# contender-order/build_game tests above are pure and must keep running without it, so this skip
# is applied per-function, not as a module-level `pytestmark` (which `pytest.importorskip` would
# effectively become, since it skips the whole module at collection time).
needs_brian2 = pytest.mark.skipif(
    importlib.util.find_spec("brian2") is None, reason="brian2 not installed (sim extra)"
)


def test_contender_order_has_all_three_contenders_exactly_once():
    order = game.contender_order("will it rain tomorrow")
    assert sorted(order) == sorted(game.CONTENDER_NAMES)
    assert len(order) == 3


def test_contender_order_is_deterministic_for_the_same_key():
    a = game.contender_order("will it rain tomorrow")
    b = game.contender_order("will it rain tomorrow")
    assert a == b


def test_contender_order_varies_across_questions():
    # Not every question can land on the same order by construction (3! = 6 possibilities) --
    # over a reasonably sized sample at least two distinct orders must occur, or the "shuffle" is
    # not actually shuffling.
    orders = {tuple(game.contender_order(f"question number {i}")) for i in range(30)}
    assert len(orders) > 1


def test_real_slot_matches_the_index_of_real_in_order():
    assert game.real_slot(["real", "random_graph", "no_brain"]) == "A"
    assert game.real_slot(["random_graph", "real", "no_brain"]) == "B"
    assert game.real_slot(["random_graph", "no_brain", "real"]) == "C"


def _game_inputs(**overrides):
    base = {
        "random_graph_trials": [(60, 40), (55, 45), (58, 42)],
        "random_graph_b0": 0.0,
        "no_brain_bias": 0.1,
        "no_brain_b0": 0.0,
    }
    base.update(overrides)
    return game.GameInputs(**base)


def test_build_game_schema_has_order_and_all_three_contenders():
    result = game.build_game(
        left_is_yes=True,
        question_key="q:will it rain",
        real_answer="yes",
        real_corrected_bias=0.15,
        real_confidence=0.75,
        game_inputs=_game_inputs(),
    )
    assert set(result.keys()) == {"order", "contenders"}
    assert sorted(result["order"]) == sorted(game.CONTENDER_NAMES)
    assert set(result["contenders"].keys()) == set(game.CONTENDER_NAMES)
    for name in game.CONTENDER_NAMES:
        c = result["contenders"][name]
        assert c["answer"] in ("yes", "no")
        assert 0.5 <= c["decisiveness"] <= 1.0
        assert isinstance(c["corrected_bias"], float)


def test_build_game_order_matches_contender_order_for_the_same_key():
    result = game.build_game(
        left_is_yes=True,
        question_key="q:will it rain",
        real_answer="yes",
        real_corrected_bias=0.15,
        real_confidence=0.75,
        game_inputs=_game_inputs(),
    )
    assert result["order"] == game.contender_order("q:will it rain")


def test_build_game_real_contender_reuses_the_already_computed_real_answer_verbatim():
    result = game.build_game(
        left_is_yes=False,
        question_key="q:x",
        real_answer="no",
        real_corrected_bias=-0.22,
        real_confidence=0.61,
        game_inputs=_game_inputs(),
    )
    real = result["contenders"]["real"]
    assert real == {"answer": "no", "corrected_bias": -0.22, "decisiveness": 0.61}


def test_build_game_random_graph_uses_its_own_b0_not_the_reals():
    trials = [(80, 20), (80, 20), (80, 20)]  # raw bias = (80-20)/100 = 0.6
    low_b0 = game.build_game(
        left_is_yes=True, question_key="q:a", real_answer="yes", real_corrected_bias=0.1,
        real_confidence=0.6, game_inputs=_game_inputs(random_graph_trials=trials, random_graph_b0=0.0),
    )
    high_b0 = game.build_game(
        left_is_yes=True, question_key="q:a", real_answer="yes", real_corrected_bias=0.1,
        real_confidence=0.6, game_inputs=_game_inputs(random_graph_trials=trials, random_graph_b0=0.5),
    )
    # Same raw trials, different b0 -> different corrected bias (subtracting a much larger
    # handedness knocks the strongly-left-biased raw reading down towards/through zero).
    rg_low = low_b0["contenders"]["random_graph"]["corrected_bias"]
    rg_high = high_b0["contenders"]["random_graph"]["corrected_bias"]
    assert rg_low == pytest.approx(0.6, abs=1e-3)
    assert rg_high == pytest.approx(0.1, abs=1e-3)


def test_build_game_random_graph_all_zero_spike_trials_is_a_neutral_tie():
    result = game.build_game(
        left_is_yes=True, question_key="q:a", real_answer="yes", real_corrected_bias=0.1,
        real_confidence=0.6,
        game_inputs=_game_inputs(random_graph_trials=[(0, 0), (0, 0), (0, 0)], random_graph_b0=0.2),
    )
    rg = result["contenders"]["random_graph"]
    assert rg["corrected_bias"] == 0.0
    assert rg["decisiveness"] == 0.5


def test_build_game_no_brain_none_bias_is_a_neutral_tie():
    result = game.build_game(
        left_is_yes=True, question_key="q:a", real_answer="yes", real_corrected_bias=0.1,
        real_confidence=0.6, game_inputs=_game_inputs(no_brain_bias=None, no_brain_b0=0.3),
    )
    nb = result["contenders"]["no_brain"]
    assert nb["corrected_bias"] == 0.0
    assert nb["decisiveness"] == 0.5


def test_build_game_no_brain_reads_the_yes_side_coin_like_every_other_contender():
    # Same raw bias/b0, opposite `left_is_yes` -> opposite answer (the coin is shared across
    # contenders, task brief: "the yes-side coin is per question, so it is shared").
    left_yes = game.build_game(
        left_is_yes=True, question_key="q:a", real_answer="yes", real_corrected_bias=0.1,
        real_confidence=0.6, game_inputs=_game_inputs(no_brain_bias=0.2, no_brain_b0=0.0),
    )
    right_yes = game.build_game(
        left_is_yes=False, question_key="q:a", real_answer="yes", real_corrected_bias=0.1,
        real_confidence=0.6, game_inputs=_game_inputs(no_brain_bias=0.2, no_brain_b0=0.0),
    )
    assert left_yes["contenders"]["no_brain"]["answer"] != right_yes["contenders"]["no_brain"]["answer"]


# -- ErContenderPool: real subprocess, tiny synthetic graph, numpy codegen (module docstring) ----


def _tiny_ring_graph(n_neurons: int = 20):
    """A small ring (matches `test_lif_record_spike_times.py`'s own fixture shape) -- deterministic
    enough to prove the pickling/reduction round trip without needing a real connectome."""
    pre_idx = np.arange(n_neurons, dtype=np.int64)
    post_idx = (pre_idx + 1) % n_neurons
    weight = np.full(n_neurons, 30.0)
    left_idx = np.array([0], dtype=np.int64)
    right_idx = np.array([1], dtype=np.int64)
    return n_neurons, pre_idx, post_idx, weight, left_idx, right_idx


@needs_brian2
def test_er_contender_pool_submit_trials_runs_a_real_subprocess_and_reduces_left_right():
    n_neurons, pre_idx, post_idx, weight, left_idx, right_idx = _tiny_ring_graph()
    pool = game.ErContenderPool(n_neurons, pre_idx, post_idx, weight, left_idx, right_idx, codegen_target="numpy")
    try:
        dense_idx = np.array([0], dtype=np.int64)
        rate_hz = np.array([300.0])
        future = pool.submit_trials(dense_idx, rate_hz, duration_ms=20.0, seeds=[1, 2, 3])
        pairs = future.result(timeout=30.0)
        assert len(pairs) == 3
        for left, right in pairs:
            assert isinstance(left, int)
            assert isinstance(right, int)
            assert left >= 0 and right >= 0
    finally:
        pool.shutdown()


@needs_brian2
def test_er_contender_pool_warm_up_succeeds_on_a_healthy_pool():
    n_neurons, pre_idx, post_idx, weight, left_idx, right_idx = _tiny_ring_graph()
    pool = game.ErContenderPool(n_neurons, pre_idx, post_idx, weight, left_idx, right_idx, codegen_target="numpy")
    try:
        pool.warm_up(timeout_s=30.0)  # must not raise
    finally:
        pool.shutdown()


@needs_brian2
def test_er_contender_pool_broken_initializer_raises_er_contender_unavailable():
    """A genuine `BrokenProcessPool`, not a mock: `post_idx` references a neuron index outside
    `n_neurons`, so `LIFNetwork.__init__`'s `Synapses.connect()` raises INSIDE the subprocess's
    initializer -- Python's own documented behaviour is that this poisons the whole pool (every
    pending/future submission raises `BrokenProcessPool`). `warm_up` must convert that into
    `ErContenderUnavailable`, exactly what `worker.build_resources`'s startup try/except and
    `worker.answer_question`'s runtime handling both key off."""
    n_neurons = 20
    pre_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    post_idx = np.array([1, 2, 3, 999], dtype=np.int64)  # 999 is out of range for n_neurons=20
    weight = np.array([5.0, 5.0, 5.0, 5.0])
    left_idx = np.array([4], dtype=np.int64)
    right_idx = np.array([5], dtype=np.int64)

    pool = game.ErContenderPool(n_neurons, pre_idx, post_idx, weight, left_idx, right_idx, codegen_target="numpy")
    try:
        with pytest.raises(game.ErContenderUnavailable):
            pool.warm_up(timeout_s=30.0)
    finally:
        pool.shutdown()


def test_expected_er_timeout_s_scales_with_trials_and_duration():
    short = game.expected_er_timeout_s(duration_ms=250.0, n_trials=3)
    longer = game.expected_er_timeout_s(duration_ms=1000.0, n_trials=3)
    assert longer > short
    assert short >= 30.0  # floor, so a tiny warm-up trial still gets a real chance to answer
    more_trials = game.expected_er_timeout_s(duration_ms=250.0, n_trials=10)
    assert more_trials > short


# -- kill(): a REAL still-running subprocess, not a fake future (round-2 logic review R1: "add a
# test with a REAL hanging subprocess ... The current timeout test uses a fake future, which is
# why it passes") -------------------------------------------------------------------------------


@needs_brian2
def test_er_contender_pool_kill_terminates_a_genuinely_still_running_child():
    n_neurons, pre_idx, post_idx, weight, left_idx, right_idx = _tiny_ring_graph()
    pool = game.ErContenderPool(n_neurons, pre_idx, post_idx, weight, left_idx, right_idx, codegen_target="numpy")
    pool.warm_up(timeout_s=30.0)
    processes = list(pool._pool._processes.values())
    assert processes and all(p.is_alive() for p in processes)

    # A long-running trial (20s of simulated time x 10 trials) so it is GENUINELY still executing
    # when the short timeout below fires -- not finished, not a mock.
    dense_idx = np.array([0], dtype=np.int64)
    rate_hz = np.array([300.0])
    future = pool.submit_trials(dense_idx, rate_hz, duration_ms=20000.0, seeds=list(range(10)))
    with pytest.raises(TimeoutError):
        future.result(timeout=0.3)
    assert all(p.is_alive() for p in processes)  # confirms it was truly still running, not a fluke

    pool.kill()
    for p in processes:
        p.join(timeout=5.0)
        assert not p.is_alive()


# The R1 bug is specifically about the WHOLE PARENT PROCESS not exiting (Python's
# `concurrent.futures` atexit hook joins the executor's manager thread, which waits on the
# child's still-running work item, forever) -- that can only be reproduced faithfully in a real
# separate process, not by asserting anything inside THIS pytest process (raising SystemExit here
# would tear down the test runner, and an in-process repro would not exercise the same atexit
# registration/ordering a real script does). Mirrors the review's own reproduction method.
_HANG_AND_EXIT_SCRIPT = """
import sys
import numpy as np
from bioreservoir.live import game

n = 20
pre_idx = np.arange(n, dtype=np.int64)
post_idx = (pre_idx + 1) % n
weight = np.full(n, 30.0)
left_idx = np.array([0], dtype=np.int64)
right_idx = np.array([1], dtype=np.int64)

pool = game.ErContenderPool(n, pre_idx, post_idx, weight, left_idx, right_idx, codegen_target="numpy")
pool.warm_up(timeout_s=30.0)
future = pool.submit_trials(np.array([0], dtype=np.int64), np.array([300.0]), duration_ms=20000.0, seeds=list(range(10)))
try:
    future.result(timeout=0.3)
except Exception:
    pass
{kill_call}
print("reached SystemExit", flush=True)
sys.exit(1)
"""


@needs_brian2
def test_worker_process_exits_promptly_after_killing_a_hung_er_child(tmp_path):
    """The actual regression: a separate Python process that hits an ER timeout with a
    GENUINELY-still-running child must exit within a few seconds once it calls `pool.kill()`
    before `sys.exit(1)` -- reproduced (round-2 review): without `kill()`, the SAME script hangs
    past a 15s `timeout` (killed with SIGTERM, exit code -15/124), because `concurrent.futures`'
    own atexit hook joins the executor's manager thread forever waiting on that running future."""
    import subprocess
    import sys
    import time

    script = tmp_path / "hang_and_exit.py"
    script.write_text(_HANG_AND_EXIT_SCRIPT.format(kill_call="pool.kill()"))

    t0 = time.monotonic()
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=15.0, check=False)
    elapsed = time.monotonic() - t0

    assert "reached SystemExit" in result.stdout, result.stderr
    assert result.returncode == 1
    assert elapsed < 10.0, f"worker process took {elapsed:.1f}s to exit -- kill() did not prevent the atexit hang"
