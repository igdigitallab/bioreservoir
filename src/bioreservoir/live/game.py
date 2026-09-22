"""The "which one is the real fly?" game (2026-09-18, following the public-launch virality
critique: seven external reviewers found the live page's real brain answers near 50/50 while the
random-graph control looks MORE confident -- instead of hiding that, this makes it the honest
headline finding rather than something to hide).

Every live question is answered by three contenders under IDENTICAL conditions (same encoded
question, same per-trial Poisson seeds, same per-question yes-side coin, same handedness-
correction formula -- each contender against its OWN measured handedness b0, never the real
brain's):

    "real"          -- the live page's existing MaleCNS LIFNetwork (worker.py, unchanged).
    "random_graph"  -- an Erdos-Renyi control with the same neuron/edge count
                       (`oracle.controls.erdos_renyi_like`), the EXACT graph the sealed batch
                       scores against (same cached file, same seed -- `oracle.config.yaml`'s
                       `seed.er_base`, see `worker.build_resources`).
    "no_brain"      -- `oracle.controls.no_brain_baseline`: the readout rule applied directly to
                       the encoder's own left/right input rates, no connectome at all.

A visitor sees three anonymous slots (A/B/C, order shuffled per-question -- `contender_order`) and
guesses which one is the real brain; `live.api`'s `POST /api/guess` scores the guess against
`Answer.game.order` (see `real_slot`).

The random graph needs a real Brian2 simulation, which would double per-question wall time if run
sequentially after the real brain's own trials. `ErContenderPool` runs it in a SEPARATE OS process
(spawned once at worker startup, one persistent `LIFNetwork` reused across every question -- the
same store/restore convention `sim.lif.LIFNetwork` itself uses, just in its own process) so it
overlaps the real brain's trials running in the worker's main process -- see docs/LIVE.md's "The
game" section for the measured timing and the `mem_limit` this needs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass

import numpy as np

from bioreservoir.live.verdict import answer_yes_no, confidence_from_bias
from bioreservoir.oracle import handedness
from bioreservoir.oracle.seeding import stable_seed

logger = logging.getLogger(__name__)


class ErContenderUnavailable(RuntimeError):
    """Raised when the random-graph control's subprocess is broken (a dead/OOM-killed child) or
    its trials did not finish within the allotted timeout (2026-09-19 logic review, F3/F4: a
    lazily-forked `ProcessPoolExecutor` that dies or hangs otherwise fails EVERY later question
    forever with a healthcheck that still reports healthy -- `pgrep` sees the still-alive parent).
    `worker.run_loop` treats this as FATAL for a question already in flight: unlike an ordinary
    per-question bug, this means the pool itself needs to be rebuilt, so the worker process exits
    non-zero instead of quietly serving "internal error" to the public forever. `worker.
    build_resources`, by contrast, catches this at STARTUP (pool build/warm-up) and disables the
    game for this process's lifetime instead of crash-looping -- see that function's docstring."""

CONTENDER_NAMES: tuple[str, ...] = ("real", "random_graph", "no_brain")
SLOTS: tuple[str, ...] = ("A", "B", "C")

# Distinct seed namespace (same convention as live/config.py's LIVE_SIDE_BASE and worker.py's
# LIVE_TRIAL_SEED_BASE) -- the per-question A/B/C shuffle must not collide with any other seeded
# coin in this pipeline.
GAME_ORDER_SEED_BASE = 20260918779


def contender_order(question_key: str) -> list[str]:
    """Deterministic per-question shuffle of CONTENDER_NAMES into A/B/C slots -- seeded only by
    the (already-normalized) question key, so asking the identical wording twice always gets the
    identical order (matching "same question -> same answer" elsewhere in this pipeline). There is
    no cross-question pattern ("A is always the real brain") a repeat asker could learn by trying
    different wordings, since each question's shuffle is seeded independently."""
    rng = np.random.default_rng(stable_seed("game-order", question_key, base=GAME_ORDER_SEED_BASE))
    order = list(CONTENDER_NAMES)
    rng.shuffle(order)
    return order


def real_slot(order: Sequence[str]) -> str:
    """Which of "A"/"B"/"C" the real brain sits in, for this question's `order` (`live.api`'s
    `POST /api/guess` scores a visitor's pick against this)."""
    return SLOTS[list(order).index("real")]


@dataclass(frozen=True)
class ContenderResult:
    answer: str  # "yes" | "no"
    corrected_bias: float
    decisiveness: float  # 0.5..1.0, same scale as the real answer's own top-level `confidence`

    def to_dict(self) -> dict:
        return {"answer": self.answer, "corrected_bias": self.corrected_bias, "decisiveness": self.decisiveness}


def _contender_result(raw_bias: float | None, b0: float, left_is_yes: bool) -> ContenderResult:
    p_yes, corrected = handedness.apply_correction(raw_bias, b0, left_is_yes, zero_spike_probability=0.5)
    safe_bias = corrected if corrected is not None else 0.0
    return ContenderResult(
        answer=answer_yes_no(p_yes), corrected_bias=safe_bias, decisiveness=confidence_from_bias(corrected)
    )


@dataclass(frozen=True)
class GameInputs:
    """Everything `build_game` needs beyond what `pipeline.compute_answer` already computed for
    the real brain -- deliberately plain numbers/tuples, no LIFNetwork/Brian2 object anywhere, so
    `pipeline.py` stays importable/testable with zero Brian2 dependency (same convention as
    `pipeline.LiveTrial`)."""

    random_graph_trials: list[tuple[int, int]]  # (left_spikes, right_spikes) per trial, SAME seeds as the real brain's own trials
    random_graph_b0: float
    no_brain_bias: float | None  # oracle.controls.no_brain_baseline's own raw bias (None if both input sides summed to 0)
    no_brain_b0: float


def build_game(
    left_is_yes: bool,
    question_key: str,
    real_answer: str,
    real_corrected_bias: float,
    real_confidence: float,
    game_inputs: GameInputs,
) -> dict:
    """`Answer.game` (docs/LIVE.md). `left_is_yes` and `question_key` are the SAME values
    `pipeline.compute_answer` already derived for the real brain's own answer (its per-question
    yes-side coin and normalized question text) -- reused here rather than recomputed, per the
    task brief: "the yes-side coin is per question, so it is shared"."""
    from bioreservoir.oracle.readout import aggregate_trials, trial_readout

    real = ContenderResult(answer=real_answer, corrected_bias=real_corrected_bias, decisiveness=real_confidence)

    rg_trials = [trial_readout(left, right) for left, right in game_inputs.random_graph_trials]
    rg_agg = aggregate_trials(rg_trials, left_is_yes=True, zero_spike_probability=0.5)
    rg_raw_bias = None if rg_agg.all_trials_zero_spikes else rg_agg.mean_bias
    random_graph = _contender_result(rg_raw_bias, game_inputs.random_graph_b0, left_is_yes)

    no_brain = _contender_result(game_inputs.no_brain_bias, game_inputs.no_brain_b0, left_is_yes)

    contenders = {"real": real, "random_graph": random_graph, "no_brain": no_brain}
    order = contender_order(question_key)
    return {"order": order, "contenders": {name: contenders[name].to_dict() for name in CONTENDER_NAMES}}


# -- random-graph control: a persistent LIFNetwork in a SEPARATE process (module docstring) --------

# Set once per subprocess by `_er_worker_init`; never touched in the parent process. Module-level
# so `_er_worker_run_trials` (also required to be a top-level, picklable function for
# `ProcessPoolExecutor`) can reach it without passing a LIFNetwork through IPC.
_ER_WORKER_NET = None


def _er_worker_init(
    n_neurons: int, pre_idx: np.ndarray, post_idx: np.ndarray, weight: np.ndarray, codegen_target: str | None = None
) -> None:
    """Runs once in the pool's single worker process (`ProcessPoolExecutor(initializer=...)`) --
    builds the Erdos-Renyi control's own `LIFNetwork` (module docstring: same store/restore
    convention as the real brain's, just in a different process so the two run concurrently)."""
    global _ER_WORKER_NET
    from bioreservoir.sim.lif import LIFNetwork

    _ER_WORKER_NET = LIFNetwork(
        n_neurons, pre_idx, post_idx, weight, record_spike_times=False, codegen_target=codegen_target
    )


def _er_worker_run_trials(
    dense_idx: np.ndarray,
    rate_hz: np.ndarray,
    duration_ms: float,
    seeds: list[int],
    readout_left_idx: np.ndarray,
    readout_right_idx: np.ndarray,
) -> list[tuple[int, int]]:
    """Runs in the worker process against `_ER_WORKER_NET` -- returns only the reduced
    `(left_spikes, right_spikes)` per trial, not the full spike-count array, to keep the IPC
    payload tiny on the per-question hot path (every question, not just the one-time b0 startup
    computation)."""
    out: list[tuple[int, int]] = []
    for seed in seeds:
        result = _ER_WORKER_NET.run_trial((dense_idx, rate_hz), duration_ms=duration_ms, seed=seed)
        counts = np.asarray(result.spike_counts)
        out.append((int(counts[readout_left_idx].sum()), int(counts[readout_right_idx].sum())))
    return out


# docs/MODEL.md's own CPU benchmark: MaleCNS-scale networks measure ~27.65s wall time per 1000ms
# simulated ("cost tracks neurons, not synapses" -- the ER control has the SAME neuron count as
# the real graph, so it costs the same per trial). Used to derive a generous per-question timeout
# for `er_future.result(...)` (F4: "e.g. 3x the expected trial time") without hardcoding a magic
# second count that would silently go stale if `config.yaml`'s trial duration ever changes.
WALL_S_PER_SIM_S = 27.65
ER_TIMEOUT_MULTIPLIER = 3.0


def expected_er_timeout_s(duration_ms: float, n_trials: int, multiplier: float = ER_TIMEOUT_MULTIPLIER) -> float:
    """`multiplier` x the expected wall time for `n_trials` trials of `duration_ms` each, per the
    benchmark above -- the value `worker.answer_question` passes to `er_future.result(timeout=...)`
    (F4). Never below 30s, so a very short `duration_ms` (e.g. a warm-up trial) still gives a
    process that is merely slow to start (not hung) a real chance to answer."""
    expected_s = (duration_ms / 1000.0) * WALL_S_PER_SIM_S * n_trials
    return max(30.0, multiplier * expected_s)


class ErContenderPool:
    """Owns the persistent, single-worker-process pool holding the ER control's `LIFNetwork`.

    `submit_trials` is non-blocking (task brief: "run it concurrently... e.g. a second process");
    `worker.answer_question` submits it BEFORE running the real brain's own trials in the main
    process, then calls `.result()` on the returned future once the real trials are done, so the
    two overlap instead of running back to back.

    Duck-typed rather than a `LIFNetwork` subclass so tests can substitute a trivial fake with the
    same `submit_trials(...) -> object with .result()` shape -- no multiprocessing/Brian2 involved
    in any test.

    `codegen_target`, if given, is forwarded to the subprocess's own `LIFNetwork` build (`numpy`
    skips the one-time C++/cython compile) -- production leaves this `None` (auto-detect `cython`,
    same as every other `LIFNetwork` in this codebase), tests pass `"numpy"` so a real, tiny,
    end-to-end subprocess test (pickling, the readout reduction, a broken-pool path) runs in
    about a second instead of paying a compile.
    """

    def __init__(
        self,
        n_neurons: int,
        pre_idx: np.ndarray,
        post_idx: np.ndarray,
        weight: np.ndarray,
        readout_left_idx: np.ndarray,
        readout_right_idx: np.ndarray,
        codegen_target: str | None = None,
    ):
        self._pool = ProcessPoolExecutor(
            max_workers=1,
            initializer=_er_worker_init,
            initargs=(n_neurons, pre_idx, post_idx, weight, codegen_target),
        )
        self._readout_left_idx = readout_left_idx
        self._readout_right_idx = readout_right_idx

    def submit_trials(self, dense_idx: np.ndarray, rate_hz: np.ndarray, duration_ms: float, seeds: list[int]) -> Future:
        """Raises `ErContenderUnavailable` immediately if the pool is already broken (a dead child
        makes `submit` itself raise `BrokenProcessPool` synchronously, before any future is even
        returned) -- callers do not need to separately guard the submit call."""
        try:
            return self._pool.submit(
                _er_worker_run_trials, dense_idx, rate_hz, duration_ms, list(seeds),
                self._readout_left_idx, self._readout_right_idx,
            )
        except BrokenProcessPool as exc:
            raise ErContenderUnavailable(f"ER control pool is broken (submit failed): {exc!r}") from exc

    def warm_up(self, timeout_s: float = 180.0) -> None:
        """Forces the pool's single worker process to spawn and build its `LIFNetwork` NOW
        (F5: "so failures are immediate") instead of lazily on the first real question --
        `ProcessPoolExecutor` forks lazily, at the first `submit`, not at construction (verified
        during the 2026-09-19 logic review: zero children exist right after `__init__`). Submits a
        trivial 1-neuron, 1ms trial and blocks for it. Raises `ErContenderUnavailable` on a
        timeout or a broken pool -- `worker.build_resources` catches this and disables the game
        for the process's lifetime rather than crash-looping (its own docstring)."""
        dummy_idx = np.zeros(1, dtype=np.int64)
        dummy_rate = np.zeros(1, dtype=np.float64)
        future = self.submit_trials(dummy_idx, dummy_rate, duration_ms=1.0, seeds=[0])
        try:
            future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            raise ErContenderUnavailable(f"ER control did not warm up within {timeout_s:.0f}s: {exc!r}") from exc
        except BrokenProcessPool as exc:
            raise ErContenderUnavailable(f"ER control pool broke during warm-up: {exc!r}") from exc

    def kill(self) -> None:
        """Force-terminate every live child process (round-2 logic review R1): a hung child
        (the documented fork-with-threads deadlock, or a genuinely wedged Brian2/cython call)
        keeps its running work item forever, and `concurrent.futures`' own `atexit` hook
        (`_python_exit`) joins the executor's manager thread before the interpreter is allowed to
        exit -- `shutdown(wait=False, ...)` alone does NOT prevent that join, so `raise
        SystemExit(1)` after a plain `shutdown()` still hangs the whole process instead of
        actually exiting (reproduced on Python 3.12 and 3.13, verified fix exits in ~3s).
        `multiprocessing.Process.kill()` sends SIGKILL directly, bypassing whatever the child is
        stuck in. Callers (`worker.run_loop`'s ER-unavailable handler, `worker.build_resources`'s
        startup except branch) call this BEFORE `raise SystemExit(1)` / before returning with the
        game disabled -- either way, no hung child should outlive the decision to stop using it.
        """
        # `_processes` (pid -> Process) is ProcessPoolExecutor's own internal bookkeeping -- there
        # is no public API to reach a running child to kill it (this is exactly why a hung child
        # needs this at all: the public `shutdown()` has no way to force one).
        for process in list(self._pool._processes.values()):
            if process.is_alive():
                process.kill()
        for process in list(self._pool._processes.values()):
            process.join(timeout=5.0)
        self._pool.shutdown(wait=False, cancel_futures=True)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def compute_er_b0_via_pool(pool, id_to_dense: dict[int, int], cfg, reference_trial_seed_fn, encoder_model=None) -> float:
    """Mirrors `worker.compute_b0_locally`'s approach ("like compute_b0_locally", task brief) but
    against the ER pool instead of the real net: run every `reference.yaml` sentence's
    `cfg.trial.n_trials` trials on the SAME already-built ER network (reused across sentences via
    `pool.submit_trials`, blocking on each -- this only runs once, at worker startup, so there is
    nothing to overlap it with), average like `oracle.handedness.compute_b0`.

    `reference_trial_seed_fn(reference_id, trial_index) -> int` is passed in rather than imported,
    so this module never depends on `worker.py` (which imports `game.py`, not the other way --
    avoids a circular import)."""
    from bioreservoir.live import config as live_config
    from bioreservoir.oracle import reference as reference_mod
    from bioreservoir.oracle.encode import encode_question_to_input
    from bioreservoir.oracle.readout import aggregate_trials, trial_readout

    sentences = reference_mod.load_reference(cfg.handedness.reference_path())
    mean_biases = []
    for sentence in sentences:
        encoded = encode_question_to_input(
            question=sentence.text,
            context="",
            dataset=live_config.LIVE_BRAIN,
            id_to_dense=id_to_dense,
            input_population=cfg.input.population,
            projection_seed=cfg.seed.projection,
            balance_seed=cfg.seed.balance,
            rate_hz_min=cfg.input.rate_hz_min,
            rate_hz_max=cfg.input.rate_hz_max,
            n_per_side=cfg.input.n_per_side,
            model=encoder_model,
        )
        seeds = [reference_trial_seed_fn(sentence.id, t) for t in range(cfg.trial.n_trials)]
        pairs = pool.submit_trials(encoded.dense_idx, encoded.rate_hz, cfg.trial.duration_ms, seeds).result()
        trials = [trial_readout(left, right) for left, right in pairs]
        agg = aggregate_trials(trials, left_is_yes=True, zero_spike_probability=0.5)
        mean_biases.append(agg.mean_bias)
    return handedness.compute_b0(mean_biases)
