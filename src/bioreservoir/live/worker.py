"""`python -m bioreservoir.live.worker` — the live fly page's real-brain loop.

Loads the male brain (MaleCNS, harmonized, `min_syn` from `experiments/001-fly-oracle/config.yaml`
— the SAME file `oracle.runner` uses for the batch pipeline, not a live-specific copy, so this
worker's `config_hash` matches the main ledger's and its own b0 lookup, if any, lines up exactly)
ONCE, builds one `LIFNetwork` with `record_spike_times=True` (for `live.frames`), then polls
`live.store.LiveStore` for queued questions FIFO: encode -> `LIVE_N_TRIALS` x `trial.duration_ms`
trials on the real network -> `live.pipeline.compute_answer`.

Must run inside the memory/CPU cage described in experiments/001-fly-oracle/RUNNING.md (same convention as
`oracle.runner`/`sim.bench`) — this module does not build the cage itself, it assumes it is
already inside one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bioreservoir.live import atlas, config, moderation, pipeline, states, store
from bioreservoir.live import game as game_mod
from bioreservoir.live import lab as lab_mod
from bioreservoir.oracle import config as oracle_config
from bioreservoir.oracle import handedness, reference
from bioreservoir.oracle.readout import aggregate_trials, trial_readout
from bioreservoir.oracle.seeding import stable_seed

# "MaleCNS v1.0" (task brief's Answer.lab.provenance.brain) -- the live page never runs BANC
# (config.LIVE_BRAIN), so this is a constant, not a per-question lookup.
BRAIN_DISPLAY_NAME = "MaleCNS v1.0"

logger = logging.getLogger(__name__)

# Distinct seed namespace from `oracle.seeding.trial_seed`/`reference_trial_seed` — live trials
# are neither a scored question's trial nor a handedness-reference trial, and must never collide
# with either (same reasoning as `config.LIVE_SIDE_BASE`).
LIVE_TRIAL_SEED_BASE = 20260918778


def live_trial_seed(question: str, trial_index: int) -> int:
    """Keyed by the normalized question text (`pipeline.question_key`), not the queue row id:
    identical questions replay identical Poisson input and so give identical answers."""
    return stable_seed("q", pipeline.question_key(question), str(trial_index), base=LIVE_TRIAL_SEED_BASE) % 2**32


def live_reference_trial_seed(reference_id: str, trial_index: int) -> int:
    return stable_seed("live-ref", reference_id, str(trial_index), base=LIVE_TRIAL_SEED_BASE) % 2**32


# -- b0: read-only from the main ledger, else computed locally and cached ------------------------


def read_only_reference_mean_biases(
    ledger_path: Path, brain: str, condition: str, config_hash: str
) -> list[float]:
    """Same query as `oracle.ledger.Ledger.reference_mean_biases`, but opened with SQLite's own
    `mode=ro` URI flag — never instantiates `Ledger` (its constructor issues `PRAGMA
    journal_mode = WAL` + `CREATE TABLE IF NOT EXISTS`, both writes) against the production
    batch's live file. Returns `[]` if the file does not exist or the query fails for any reason
    a missing/mid-write database could cause (`sqlite3.OperationalError`) — the caller falls back
    to computing b0 locally either way, so "no usable reference runs yet" and "can't even open
    the file" are handled identically.
    """
    if not ledger_path.exists():
        return []
    uri = f"file:{ledger_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            rows = conn.execute(
                "SELECT p.mean_bias FROM reference_runs r JOIN reference_predictions p "
                "ON p.run_id = r.run_id WHERE r.status = 'done' AND r.brain = ? AND "
                "r.condition = ? AND r.config_hash = ?",
                (brain, condition, config_hash),
            ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()
    except sqlite3.OperationalError:
        logger.warning("could not read main ledger read-only at %s", ledger_path, exc_info=True)
        return []


def compute_b0_locally(net, id_to_dense: dict[int, int], cfg, readout_left_idx, readout_right_idx, encoder_model=None) -> float:
    """Fallback when the main ledger has no `(malecns, real)` reference runs yet under this
    `config_hash`: run `reference.yaml`'s 24 sentences through the ALREADY-BUILT live network
    (reused, not a second `LIFNetwork`), `cfg.trial.n_trials` trials each, exactly like
    `oracle.runner.run_sim_group`'s own reference-item loop, and average (`oracle.handedness.
    compute_b0`). `net` only needs a `.run_trial(inputs, duration_ms, seed) -> result` method
    (`result.spike_counts` indexable by the readout index arrays) — duck-typed so this function
    is unit-testable with a fake network, no Brian2 required.
    """
    from bioreservoir.oracle.encode import encode_question_to_input

    sentences = reference.load_reference(cfg.handedness.reference_path())
    mean_biases = []
    for sentence in sentences:
        encoded = encode_question_to_input(
            question=sentence.text,
            context="",
            dataset=config.LIVE_BRAIN,
            id_to_dense=id_to_dense,
            input_population=cfg.input.population,
            projection_seed=cfg.seed.projection,
            balance_seed=cfg.seed.balance,
            rate_hz_min=cfg.input.rate_hz_min,
            rate_hz_max=cfg.input.rate_hz_max,
            n_per_side=cfg.input.n_per_side,
            model=encoder_model,
        )
        trials = []
        for t in range(cfg.trial.n_trials):
            seed = live_reference_trial_seed(sentence.id, t)
            result = net.run_trial(
                (encoded.dense_idx, encoded.rate_hz), duration_ms=cfg.trial.duration_ms, seed=seed
            )
            left = int(np.asarray(result.spike_counts)[readout_left_idx].sum())
            right = int(np.asarray(result.spike_counts)[readout_right_idx].sum())
            trials.append(trial_readout(left, right))
        agg = aggregate_trials(trials, left_is_yes=True, zero_spike_probability=0.5)
        mean_biases.append(agg.mean_bias)
    return handedness.compute_b0(mean_biases)


def load_b0_cache(path: Path, brain: str, condition: str, config_hash: str) -> float | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    key = f"{brain}:{condition}:{config_hash}"
    entry = cached.get(key)
    return entry["b0"] if entry is not None else None


def save_b0_cache(path: Path, brain: str, condition: str, config_hash: str, b0: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}
    key = f"{brain}:{condition}:{config_hash}"
    cached[key] = {"b0": b0, "computed_at": time.time()}
    path.write_text(json.dumps(cached, indent=2, sort_keys=True))


def _b0_from_ledger_or_cache(cfg_hash: str, condition: str) -> float | None:
    """The first two steps of the task brief's b0 order (main ledger read-only, then the local
    JSON cache), shared by `get_or_compute_b0` (condition="real") and the game's own
    `get_or_compute_er_b0`/`get_or_compute_no_brain_b0` below — `None` if neither has it yet,
    meaning the caller must compute it locally."""
    biases = read_only_reference_mean_biases(config.MAIN_LEDGER_PATH, config.LIVE_BRAIN, condition, cfg_hash)
    if biases:
        b0 = handedness.compute_b0(biases)
        logger.info("b0 loaded from main ledger (%s): %.4f (n=%d reference sentences)", condition, b0, len(biases))
        return b0

    cached_b0 = load_b0_cache(config.B0_CACHE_PATH, config.LIVE_BRAIN, condition, cfg_hash)
    if cached_b0 is not None:
        logger.info("b0 loaded from local cache (%s): %.4f", condition, cached_b0)
        return cached_b0
    return None


def get_or_compute_b0(
    net, id_to_dense, cfg, cfg_hash: str, readout_left_idx, readout_right_idx, encoder_model=None,
    *, condition: str = "real",
) -> float:
    """b0 for `(malecns, condition)` under this worker's `config_hash`: main ledger (read-only)
    first, then the local JSON cache, then computed from scratch against `net` (and cached) —
    task brief order. `condition` defaults to "real" (every caller before the game feature, and
    every existing test, passes no `condition` at all) — the game's random-graph control reuses
    this exact function with `condition="er"` and its own already-built ER `net`/readout indices
    (`get_or_compute_er_b0` below), since the ledger/cache mechanics are identical."""
    b0 = _b0_from_ledger_or_cache(cfg_hash, condition)
    if b0 is not None:
        return b0

    logger.warning("no b0 in main ledger or cache (%s) — computing locally from reference.yaml", condition)
    b0 = compute_b0_locally(net, id_to_dense, cfg, readout_left_idx, readout_right_idx, encoder_model)
    save_b0_cache(config.B0_CACHE_PATH, config.LIVE_BRAIN, condition, cfg_hash, b0)
    logger.info("b0 computed locally and cached (%s): %.4f", condition, b0)
    return b0


def get_or_compute_er_b0(er_pool, id_to_dense, cfg, cfg_hash: str, encoder_model=None) -> float:
    """Same ledger -> cache -> compute order as `get_or_compute_b0`, but for the game's random-
    graph control (task brief: "the random graph needs its own b0 ... compute it once at worker
    start ... and cache it next to the real one"). The reference-sentence trials must run against
    the ER pool's OWN network, not the real `LIFNetwork` — `game.compute_er_b0_via_pool` submits
    them through `er_pool` exactly like `answer_question` submits a question's trials."""
    b0 = _b0_from_ledger_or_cache(cfg_hash, "er")
    if b0 is not None:
        return b0

    logger.warning("no b0 in main ledger or cache (er) — computing locally against the game's ER pool")
    b0 = game_mod.compute_er_b0_via_pool(er_pool, id_to_dense, cfg, live_reference_trial_seed, encoder_model)
    save_b0_cache(config.B0_CACHE_PATH, config.LIVE_BRAIN, "er", cfg_hash, b0)
    logger.info("er b0 computed locally and cached: %.4f", b0)
    return b0


def compute_no_brain_b0_locally(id_to_dense, cfg, encoder_model=None) -> float:
    """No-connectome control's own handedness (`oracle.runner.run_no_brain_group`'s convention,
    reused here for the live game): each of reference.yaml's 24 sentences' raw input-drive bias
    (`oracle.controls.no_brain_baseline`, no LIFNetwork at all), averaged. Cheap — no simulation,
    safe to run directly in the worker/API process, no cage required."""
    from bioreservoir.oracle import controls as controls_mod
    from bioreservoir.oracle.encode import encode_question_to_input

    sentences = reference.load_reference(cfg.handedness.reference_path())
    biases = []
    for sentence in sentences:
        encoded = encode_question_to_input(
            question=sentence.text,
            context="",
            dataset=config.LIVE_BRAIN,
            id_to_dense=id_to_dense,
            input_population=cfg.input.population,
            projection_seed=cfg.seed.projection,
            balance_seed=cfg.seed.balance,
            rate_hz_min=cfg.input.rate_hz_min,
            rate_hz_max=cfg.input.rate_hz_max,
            n_per_side=cfg.input.n_per_side,
            model=encoder_model,
        )
        result = controls_mod.no_brain_baseline(encoded.left_rate_hz, encoded.right_rate_hz)
        biases.append(result["bias"] if result["bias"] is not None else 0.0)
    return handedness.compute_b0(biases)


def get_or_compute_no_brain_b0(id_to_dense, cfg, cfg_hash: str, encoder_model=None) -> float:
    b0 = _b0_from_ledger_or_cache(cfg_hash, "no_brain")
    if b0 is not None:
        return b0

    logger.warning("no b0 in main ledger or cache (no_brain) — computing locally")
    b0 = compute_no_brain_b0_locally(id_to_dense, cfg, encoder_model)
    save_b0_cache(config.B0_CACHE_PATH, config.LIVE_BRAIN, "no_brain", cfg_hash, b0)
    logger.info("no_brain b0 computed locally and cached: %.4f", b0)
    return b0


# -- annotation lookups for Answer.lab (cell type / super class / modality per neuron) -----------


def build_annotation_lookups(dataset: str, id_to_dense: dict[int, int]) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """`(cell_type_lookup, super_class_lookup, modality_lookup)`, each `{dense_idx: str}` for
    every neuron in the harmonized graph — built once at startup (`build_resources`), consumed
    by `live.lab`'s pure functions, which never touch pyarrow/raw annotation files themselves.
    `modality` is the dataset's own raw per-neuron functional class (MaleCNS: `type`/`class`
    columns; BANC: `cell_type`/`cell_class`) — the most granular label the data actually has, not
    an invented bucketing (operator requirement: no decorative numbers/labels).
    """
    from bioreservoir.connectomes import populations as populations_mod

    table = populations_mod.raw_annotations(dataset)
    neuron_ids = table.column("neuron_id").to_pylist()
    if dataset == "malecns":
        cell_types = table.column("type").to_pylist()
        modalities = table.column("class").to_pylist()
    else:
        cell_types = table.column("cell_type").to_pylist()
        modalities = table.column("cell_class").to_pylist()
    super_classes = table.column("super_class").to_pylist()

    cell_type_lookup: dict[int, str] = {}
    super_class_lookup: dict[int, str] = {}
    modality_lookup: dict[int, str] = {}
    for neuron_id, cell_type, super_class, modality in zip(neuron_ids, cell_types, super_classes, modalities, strict=True):
        dense = id_to_dense.get(neuron_id)
        if dense is None:
            continue
        cell_type_lookup[dense] = cell_type or "unknown"
        super_class_lookup[dense] = super_class or "unknown"
        modality_lookup[dense] = modality or "unknown"
    return cell_type_lookup, super_class_lookup, modality_lookup


# -- answering one question -----------------------------------------------------------------------


def answer_question(
    row,
    net,
    id_to_dense: dict[int, int],
    cfg,
    b0: float,
    lab_ctx: lab_mod.LabContext,
    state_specs: dict[str, states.StateSpec],
    state_population_idx: dict[str, np.ndarray | None],
    encoder_model=None,
    *,
    er_pool=None,
    random_graph_b0: float | None = None,
    no_brain_b0: float | None = None,
) -> dict:
    """`row` (a `store.LiveStore` row: `id`, `question`, ...) -> the Answer dict
    (`pipeline.compute_answer`'s return value). `net` is duck-typed (see `compute_b0_locally`'s
    docstring) so this whole function is testable with a fake network standing in for
    `LIFNetwork` — no Brian2/full-brain simulation needed to exercise it.

    `er_pool` (a `game.ErContenderPool`, or any duck-typed fake with the same `submit_trials(...)
    -> future` shape), if given, adds the "which one is the real fly?" game (`Answer.game`):
    its trials are submitted BEFORE the real brain's own trial loop below runs, using the SAME
    per-trial seeds (`live_trial_seed`, task brief: "the same per-question Poisson seeds ...
    shared"), so the random-graph control's Brian2 run in its own process overlaps the real
    brain's run in this process instead of adding to its wall time. `er_pool=None` (every caller
    before the game feature, and every existing test) omits `Answer.game` entirely — see
    `pipeline.compute_answer`'s `game_inputs` docstring.
    """
    from bioreservoir.oracle.encode import encode_question_to_input

    question = row["question"]
    encoded = encode_question_to_input(
        question=question,
        context="",
        dataset=config.LIVE_BRAIN,
        id_to_dense=id_to_dense,
        input_population=cfg.input.population,
        projection_seed=cfg.seed.projection,
        balance_seed=cfg.seed.balance,
        rate_hz_min=cfg.input.rate_hz_min,
        rate_hz_max=cfg.input.rate_hz_max,
        n_per_side=cfg.input.n_per_side,
        model=encoder_model,
    )

    # Task brief F7 (2026-09-19 logic review): a resources dict that has an `er_pool` but is
    # missing either control's b0 (e.g. a partially-built/stale resources dict) must NOT silently
    # score those contenders as if b0=0.0 -- that is exactly the "unmeasured group through
    # uncorrected" failure `oracle.handedness.compute_b0` itself refuses to allow. Disable the
    # game for THIS answer only (log, don't raise -- a transient/malformed caller should not take
    # down an otherwise-healthy question) and serve the plain real-brain answer.
    game_enabled = er_pool is not None
    if game_enabled and (random_graph_b0 is None or no_brain_b0 is None):
        logger.error(
            "game disabled for id=%s: random_graph_b0=%r no_brain_b0=%r missing -- serving the "
            "plain real-brain answer only",
            row["id"], random_graph_b0, no_brain_b0,
        )
        game_enabled = False

    er_future = None
    if game_enabled:
        er_seeds = [live_trial_seed(question, t) for t in range(config.LIVE_N_TRIALS)]
        er_future = er_pool.submit_trials(encoded.dense_idx, encoded.rate_hz, cfg.trial.duration_ms, er_seeds)

    trials: list[pipeline.LiveTrial] = []
    for t in range(config.LIVE_N_TRIALS):
        seed = live_trial_seed(question, t)
        result = net.run_trial(
            (encoded.dense_idx, encoded.rate_hz), duration_ms=cfg.trial.duration_ms, seed=seed
        )
        spike_counts = np.asarray(result.spike_counts)
        left = int(spike_counts[lab_ctx.readout_left_idx].sum())
        right = int(spike_counts[lab_ctx.readout_right_idx].sum())
        state_counts = {
            pop: (int(spike_counts[idx].sum()) if idx is not None and idx.size else 0)
            for pop, idx in state_population_idx.items()
        }
        n_active = int((spike_counts > 0).sum())
        trials.append(
            pipeline.LiveTrial(
                left_spikes=left,
                right_spikes=right,
                seed=seed,
                total_spikes=int(spike_counts.sum()),
                state_spike_counts=state_counts,
                n_active_neurons=n_active,
                spike_neuron_idx=getattr(result, "spike_neuron_idx", None),
                spike_time_ms=getattr(result, "spike_time_ms", None),
            )
        )

    game_inputs = None
    if game_enabled:
        from concurrent.futures import TimeoutError as FutureTimeoutError
        from concurrent.futures.process import BrokenProcessPool

        from bioreservoir.oracle import controls as controls_mod

        # F4: a bounded wait, not an unbounded one -- a hung ER subprocess (e.g. the documented
        # fork-with-threads deadlock hazard) would otherwise leave this question (and the FIFO
        # queue behind it) stuck in "thinking" forever with a still-green pgrep healthcheck.
        timeout_s = game_mod.expected_er_timeout_s(cfg.trial.duration_ms, config.LIVE_N_TRIALS)
        try:
            random_graph_trials = er_future.result(timeout=timeout_s)  # blocks only until the ER process, run concurrently above, finishes
        except FutureTimeoutError as exc:
            raise game_mod.ErContenderUnavailable(
                f"ER control did not answer id={row['id']} within {timeout_s:.0f}s: {exc!r}"
            ) from exc
        except BrokenProcessPool as exc:
            raise game_mod.ErContenderUnavailable(f"ER control pool broke while answering id={row['id']}: {exc!r}") from exc
        no_brain_result = controls_mod.no_brain_baseline(encoded.left_rate_hz, encoded.right_rate_hz)
        # random_graph_b0/no_brain_b0 are guaranteed non-None here -- `game_enabled` above already
        # refused to reach this branch otherwise (F7: never a silent 0.0 fallback for a missing b0).
        assert random_graph_b0 is not None and no_brain_b0 is not None
        game_inputs = game_mod.GameInputs(
            random_graph_trials=random_graph_trials,
            random_graph_b0=random_graph_b0,
            no_brain_bias=no_brain_result["bias"],
            no_brain_b0=no_brain_b0,
        )

    return pipeline.compute_answer(
        id_=row["id"],
        question=question,
        trials=trials,
        b0=b0,
        state_specs=state_specs,
        duration_ms=cfg.trial.duration_ms,
        lab_ctx=lab_ctx,
        stimulated_dense_idx=encoded.dense_idx,
        stimulated_left_idx=encoded.left_idx,
        stimulated_right_idx=encoded.right_idx,
        game_inputs=game_inputs,
    )


# -- setup + main loop -----------------------------------------------------------------------------


class _GameOff(Exception):
    """Control flow only: skips the game's setup block when `config.LIVE_GAME_ENABLED` is off."""


def build_resources():
    """Everything the loop needs, built once: `(net, id_to_dense, cfg, cfg_hash, lab_ctx,
    state_specs, state_population_idx, encoder_model, b0, er_pool, random_graph_b0,
    no_brain_b0)`."""
    from bioreservoir.oracle import controls
    from bioreservoir.oracle import ledger as oracle_ledger
    from bioreservoir.oracle.encode import load_encoder
    from bioreservoir.sim import bench
    from bioreservoir.sim.lif import DT_MS, LIFNetwork

    cfg = oracle_config.load_config()
    cfg_hash = oracle_config.config_hash(cfg)

    logger.info("loading %s (min_syn=%d) ...", config.LIVE_BRAIN, cfg.trial.min_syn)
    n_neurons, pre_idx, post_idx, weight, id_to_dense = bench.graph_to_arrays(
        config.LIVE_BRAIN, min_syn=cfg.trial.min_syn
    )
    logger.info("building LIFNetwork (n_neurons=%d, record_spike_times=True) ...", n_neurons)
    net = LIFNetwork(n_neurons, pre_idx, post_idx, weight, record_spike_times=True)

    encoder_model = load_encoder()

    readout_left_idx = bench.population_indices(config.LIVE_BRAIN, id_to_dense, cfg.readout_population_left())
    readout_right_idx = bench.population_indices(config.LIVE_BRAIN, id_to_dense, cfg.readout_population_right())
    if readout_left_idx is None or readout_right_idx is None:
        raise ValueError(f"readout population {cfg.readout.name!r} is absent for {config.LIVE_BRAIN!r}")

    # -- game: the random-graph control, built from the EXACT same cached graph the sealed batch
    # scores against (same `min_syn`, same `cfg.seed.er_base` -- `controls.cached_control_graph`
    # reads `data/processed/malecns-min<N>-er-seed<er_base>/edges.parquet` if the batch already
    # produced it, else builds and caches it here). Its own `LIFNetwork` lives in a separate
    # process (`game.ErContenderPool`) so it can run concurrently with the real brain's trials —
    # see game.py's module docstring and docs/LIVE.md's "The game" section for the measured cost.
    #
    # Everything in this block is a STARTUP-time concern (2026-09-19 logic review, "should fix"):
    # a missing/unwritable ER cache (the `data/processed` mount is read-only in production, so a
    # cache miss cannot be built and written on the fly), a broken/timed-out warm-up, or a b0
    # computation failure must NOT crash-loop the container -- restarting fixes none of them. Log
    # loudly and disable the game for this process's lifetime instead; the plain real-brain answer
    # keeps working. Contrast with `answer_question`'s `ErContenderUnavailable`, which IS meant to
    # crash the worker (F3/F4) -- that is for a pool that started fine and later died at runtime,
    # where a restart (a fresh pool, fresh memory) is the actual fix.
    er_pool: game_mod.ErContenderPool | None = None
    random_graph_b0: float | None = None
    no_brain_b0: float | None = None
    try:
        if not config.LIVE_GAME_ENABLED:
            raise _GameOff
        logger.info("building the game's random-graph control (Erdos-Renyi, seed=%d) ...", cfg.seed.er_base)
        er_graph = controls.cached_control_graph(
            config.LIVE_BRAIN,
            cfg.trial.min_syn,
            "er",
            cfg.seed.er_base,
            build=lambda: controls.erdos_renyi_like(n_neurons, pre_idx, post_idx, weight, seed=cfg.seed.er_base),
        )
        er_pool = game_mod.ErContenderPool(
            er_graph.n_neurons, er_graph.pre_idx, er_graph.post_idx, er_graph.weight, readout_left_idx, readout_right_idx
        )
        # F5: force the subprocess to spawn and build its LIFNetwork NOW, not lazily on the first
        # real visitor's question -- a broken pool/cache fails HERE, at startup, not later.
        logger.info("warming up the game's ER control network ...")
        er_pool.warm_up(timeout_s=config.ER_WARMUP_TIMEOUT_S)
        random_graph_b0 = get_or_compute_er_b0(er_pool, id_to_dense, cfg, cfg_hash, encoder_model)
        no_brain_b0 = get_or_compute_no_brain_b0(id_to_dense, cfg, cfg_hash, encoder_model)
    except _GameOff:
        logger.info("the game is switched off (LIVE_GAME != 1): answers carry no game block")
    except Exception:
        logger.exception(
            "UNHANDLED exc_class=GameSetupFailed path=worker.build_resources -- the game's "
            "random-graph control could not be built (missing/unwritable ER cache, a broken "
            "subprocess, or a b0 computation failure). Serving every answer WITHOUT the game "
            "until this is fixed and the worker is restarted -- NOT crash-looping, since a "
            "restart alone would not fix a missing cache file or a bad mount."
        )
        if er_pool is not None:
            # R1 (round-2 logic review): a warm-up TIMEOUT means the subprocess is still running
            # (not dead) -- `shutdown(wait=False)` alone leaves that ~1.9 GB child alive for the
            # rest of this process's lifetime. `kill()` force-terminates it before moving on.
            er_pool.kill()
        er_pool = None
        random_graph_b0 = None
        no_brain_b0 = None

    state_specs = states.load_states()
    state_population_idx: dict[str, np.ndarray | None] = {}
    for spec in state_specs.values():
        for pop in spec.populations:
            if pop not in state_population_idx:
                state_population_idx[pop] = bench.population_indices(config.LIVE_BRAIN, id_to_dense, pop)

    dense_to_atlas = None
    if atlas.atlas_files_exist():
        try:
            neuron_ids = atlas.load_neuron_ids()
            dense_to_atlas = atlas.build_dense_to_atlas(id_to_dense, neuron_ids)
            logger.info("atlas loaded: %d neurons mapped", len(dense_to_atlas))
        except Exception:
            logger.exception("failed to load atlas — continuing with frames disabled")
    else:
        logger.warning("atlas files not found at %s — frames will be omitted from every answer", config.ATLAS_DIR)

    logger.info("building annotation lookups (cell_type/super_class/modality) ...")
    cell_type_lookup, super_class_lookup, modality_lookup = build_annotation_lookups(config.LIVE_BRAIN, id_to_dense)

    # `n_synapses = sum(|signed_weight|)`: `signed_weight = sign(pre) * syn_count` and `sign` is
    # always +-1 (never 0, `schema.py`'s binary Shiu rule), so `abs(signed_weight) == syn_count`
    # exactly -- no separate raw-edge-table read needed for the true synapse count (as opposed to
    # `n_connections`, the edge/connection count, already `len(pre_idx)`).
    n_connections = len(pre_idx)
    n_synapses = int(np.abs(weight).sum())
    # `git_sha()` returns `None` in this container (no `.git` — see config.CODE_SHA_FALLBACK's
    # docstring); fall back to the commit deploy.sh baked in at image build time.
    code_sha = oracle_ledger.git_sha() or config.CODE_SHA_FALLBACK

    lab_ctx = lab_mod.LabContext(
        brain_name=BRAIN_DISPLAY_NAME,
        n_neurons=n_neurons,
        n_connections=n_connections,
        n_synapses=n_synapses,
        min_syn=cfg.trial.min_syn,
        dt_ms=DT_MS,
        code_sha=code_sha,
        config_hash=cfg_hash,
        cell_type_lookup=cell_type_lookup,
        super_class_lookup=super_class_lookup,
        modality_lookup=modality_lookup,
        readout_left_idx=readout_left_idx,
        readout_right_idx=readout_right_idx,
        dense_to_atlas=dense_to_atlas,
    )

    b0 = get_or_compute_b0(net, id_to_dense, cfg, cfg_hash, readout_left_idx, readout_right_idx, encoder_model)

    return {
        "net": net,
        "id_to_dense": id_to_dense,
        "cfg": cfg,
        "cfg_hash": cfg_hash,
        "lab_ctx": lab_ctx,
        "state_specs": state_specs,
        "state_population_idx": state_population_idx,
        "encoder_model": encoder_model,
        "b0": b0,
        "er_pool": er_pool,
        "random_graph_b0": random_graph_b0,
        "no_brain_b0": no_brain_b0,
    }


def touch_heartbeat(path: Path | None = None) -> None:
    """Written on every `run_loop` iteration (idle poll or a finished question) and once right
    after `build_resources()` returns in `main()` -- proof the main loop is actually alive, not
    just that the process exists (F3/F4: a `pgrep`-style healthcheck stays green even when the
    loop is wedged inside a hung blocking call). See `config.WORKER_HEARTBEAT_PATH`'s docstring
    for the compose healthcheck command that reads this file's freshness.

    `path` defaults to `config.WORKER_HEARTBEAT_PATH`, read at CALL time (not as a default
    argument value, which would bind once at import and ignore a test's `monkeypatch.setattr
    (config, "WORKER_HEARTBEAT_PATH", ...)`)."""
    path = path if path is not None else config.WORKER_HEARTBEAT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now(UTC).isoformat())


def run_loop(resources: dict, live_store: store.LiveStore, once: bool = False) -> None:
    n_processed = 0
    while True:
        touch_heartbeat()
        row = live_store.claim_next()
        if row is None:
            if once:
                return
            time.sleep(config.LIVE_POLL_INTERVAL_S)
            continue
        # Queued while the moderation classifier was busy (`moderation.LLM_BUSY`, a crowd hitting
        # its per-minute cap): classify it now, right before spending 30 s of CPU on it. One call
        # per answered question is always inside the budget, and the asker gets the verdict on
        # their own question page instead of a "looks like spam" at ask time.
        if row["needs_llm"]:
            verdict_result = asyncio.run(moderation.moderate_content(row["question"]))
            if not verdict_result.ok:
                reason = verdict_result.reason
                message = verdict_result.message
                if reason == moderation.LLM_BUSY:
                    # Still busy after waiting in line: fail closed, and say so plainly.
                    reason, message = "spam", "We could not check this question with our moderation gate. Please ask it again."
                logger.info("moderation rejected id=%d reason=%s", row["id"], reason)
                live_store.reject_claimed(row["id"], reason, message)
                n_processed += 1
                if once:
                    return
                continue
        try:
            answer = answer_question(
                row,
                resources["net"],
                resources["id_to_dense"],
                resources["cfg"],
                resources["b0"],
                resources["lab_ctx"],
                resources["state_specs"],
                resources["state_population_idx"],
                resources["encoder_model"],
                er_pool=resources.get("er_pool"),
                random_graph_b0=resources.get("random_graph_b0"),
                no_brain_b0=resources.get("no_brain_b0"),
            )
            live_store.record_answer(row["id"], answer)
            logger.info("answered id=%d question=%r", row["id"], row["question"])
        except game_mod.ErContenderUnavailable as exc:
            # F3/F4: the random-graph control's subprocess is broken or hung -- every LATER
            # question would fail the exact same way, so this is fatal for the whole process, not
            # just this one question. Mark it failed with a retry-oriented message (not a silent
            # hang), then exit non-zero so the container's restart policy gets a fresh process and
            # a fresh pool -- `except Exception` below deliberately does NOT catch this (it is a
            # more specific handler, listed first).
            logger.exception(
                "UNHANDLED exc_class=%s path=worker.answer_question id=%s -- ER control "
                "unavailable, exiting so the container restarts with a fresh pool",
                type(exc).__name__, row["id"],
            )
            live_store.mark_failed(row["id"], "the fly's brain hit a snag — please ask again in a minute")
            # R1 (round-2 logic review): on a TIMEOUT the child is still running, not dead --
            # `SystemExit` alone does not exit the process, because `concurrent.futures`' own
            # `atexit` hook joins the executor's manager thread forever waiting for that running
            # work item. `kill()` force-terminates the child FIRST so the process actually exits
            # (reproduced hang on Python 3.12/3.13 without this; verified fix exits in ~3s).
            er_pool = resources.get("er_pool")
            if er_pool is not None:
                er_pool.kill()
            raise SystemExit(1) from exc
        except Exception as exc:
            logger.exception(
                "UNHANDLED exc_class=%s path=worker.answer_question id=%s",
                type(exc).__name__, row["id"],
            )
            live_store.mark_failed(row["id"], "internal error while answering this question")
        n_processed += 1
        if n_processed % 100 == 0:
            n_purged = live_store.purge_expired()
            if n_purged:
                logger.info("purged %d expired rejected question(s)", n_purged)
            # Chat retention rides the same cadence (security review F4 + "purge soft-deleted
            # chat after 7 days") -- chat traffic is independent of the question queue, so this
            # is an approximation (idle question queue = chat purge also stalls), same limitation
            # `purge_expired` above already has; good enough until chat volume justifies its own
            # scheduler.
            chat_purged = live_store.chat_purge_expired()
            if any(chat_purged.values()):
                logger.info(
                    "purged chat: %d text, %d identity, %d attempts, %d llm_calls",
                    chat_purged["text_blanked"],
                    chat_purged["identity_blanked"],
                    chat_purged["attempts_purged"],
                    chat_purged["llm_calls_purged"],
                )
        if once:
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once", action="store_true", help="process at most one queued question, then exit"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("live worker starting (brain=%s)", config.LIVE_BRAIN)

    resources = build_resources()
    logger.info(
        "live worker ready: b0=%.4f random_graph_b0=%s no_brain_b0=%s (game %s)",
        resources["b0"], resources["random_graph_b0"], resources["no_brain_b0"],
        "enabled" if resources["er_pool"] is not None else "DISABLED -- see the GameSetupFailed log above",
    )
    # Startup (build_resources, including the game's own warm-up) can legitimately take a while
    # (a one-time Brian2/cython compile) -- a fresh heartbeat right as the loop begins means the
    # compose healthcheck's `start_period` only has to cover build time once, not also the first
    # `run_loop` iteration.
    touch_heartbeat()

    try:
        with store.LiveStore() as live_store:
            orphans = live_store.requeue_orphans()
            if orphans:
                logger.warning("requeued %d question(s) a previous worker died on: %s", len(orphans), orphans)
            run_loop(resources, live_store, once=args.once)
    finally:
        # The game's ER control network lives in its own subprocess (game.ErContenderPool) --
        # shut its pool down on exit rather than relying only on ProcessPoolExecutor's own atexit
        # handler, so `--once` runs (tests, manual smoke checks) release it immediately. `None`
        # when build_resources() disabled the game at startup (missing cache, broken pool, ...).
        if resources["er_pool"] is not None:
            resources["er_pool"].shutdown()


if __name__ == "__main__":
    main()
