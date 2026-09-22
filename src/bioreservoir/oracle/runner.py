"""`python -m bioreservoir.oracle run` — executes the work plan (`oracle.plan`) against the real
connectomes, resumable via the ledger.

Sim-backed groups (`condition in {real, rewired, er}`) are dispatched to a
`ProcessPoolExecutor(max_workers=...)`: one process per `(brain, condition)` group builds exactly
one `LIFNetwork` (task brief: "One network build per (brain, condition) per worker, reused via
store/restore") and runs every question/variant/trial that group needs before exiting. The two
cheap controls (no-brain, coin) need no connectome and run directly in the calling process.

Must run inside the memory/CPU cage described in experiments/001-fly-oracle/RUNNING.md — this module does not build the cage
itself, same convention as `sim/bench.py` and `sim/calibrate.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bioreservoir.oracle import controls, handedness, plan, side_mapping
from bioreservoir.oracle.config import OracleConfig
from bioreservoir.oracle.config import config_hash as compute_config_hash
from bioreservoir.oracle.ledger import Ledger
from bioreservoir.oracle.plan import ReferenceWorkItem, WorkItem
from bioreservoir.oracle.questions import Question, load_questions
from bioreservoir.oracle.readout import aggregate_trials, trial_readout
from bioreservoir.oracle.reference import ReferenceSentence, load_reference
from bioreservoir.oracle.seeding import reference_trial_seed, trial_seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialSpike:
    left: int
    right: int


def build_graph_for_condition(brain: str, condition: str, config: OracleConfig):
    """`(n_neurons, pre_idx, post_idx, weight, id_to_dense)` for `condition` on `brain`.

    "real" is `bioreservoir.sim.bench.graph_to_arrays` (which itself wraps
    `harmonize.load_graph`'s own parquet cache); "rewired"/"er" derive from the real graph via
    `oracle.controls`, cached separately under `data/processed/` with the seed in the path
    (`controls.cached_control_graph`).
    """
    from bioreservoir.sim import bench

    n_neurons, pre_idx, post_idx, weight, id_to_dense = bench.graph_to_arrays(brain, min_syn=config.trial.min_syn)
    if condition == "real":
        return n_neurons, pre_idx, post_idx, weight, id_to_dense

    if condition == "rewired":
        seed = config.seed.rewire_base
        graph = controls.cached_control_graph(
            brain,
            config.trial.min_syn,
            "rewired",
            seed,
            build=lambda: controls.rewire_degree_preserving(
                n_neurons, pre_idx, post_idx, weight, seed=seed,
                max_repair_rounds=config.controls.rewire_max_repair_rounds,
            ),
        )
    elif condition == "er":
        seed = config.seed.er_base
        graph = controls.cached_control_graph(
            brain,
            config.trial.min_syn,
            "er",
            seed,
            build=lambda: controls.erdos_renyi_like(n_neurons, pre_idx, post_idx, weight, seed=seed),
        )
    else:
        raise ValueError(f"unknown sim condition: {condition!r}")
    return graph.n_neurons, graph.pre_idx, graph.post_idx, graph.weight, id_to_dense


def _readout_side_indices(brain: str, id_to_dense: dict[int, int], config: OracleConfig):
    from bioreservoir.sim import bench

    left = bench.population_indices(brain, id_to_dense, config.readout_population_left())
    right = bench.population_indices(brain, id_to_dense, config.readout_population_right())
    if left is None or right is None:
        raise ValueError(
            f"readout population {config.readout.name!r} is absent for {brain!r} "
            "(populations.yaml) — pick a different readout.name in config.yaml"
        )
    return left, right


def run_sim_group(
    brain: str,
    condition: str,
    items: list[WorkItem],
    reference_items: list[ReferenceWorkItem],
    questions: dict[str, Question],
    reference_sentences: dict[str, ReferenceSentence],
    config: OracleConfig,
    ledger: Ledger,
    cfg_hash: str,
    encoder_model=None,
) -> tuple[int, int]:
    """Process one `(brain, condition)` group: build the network once, run every pending
    handedness reference sentence (`oracle.handedness`) first, then for each (question, variant)
    pair encode, run `config.trial.n_trials` trials, subtract this group's own `b0` and map the
    corrected bias to `P(yes)` via this question's own YES-side coin
    (`oracle.side_mapping.left_is_yes_for`). Returns `(n_question_items_done, n_reference_items_done)`.
    """
    from bioreservoir.oracle.encode import encode_question_to_input
    from bioreservoir.sim.lif import LIFNetwork

    n_neurons, pre_idx, post_idx, weight, id_to_dense = build_graph_for_condition(brain, condition, config)
    net = LIFNetwork(n_neurons, pre_idx, post_idx, weight)
    readout_left_idx, readout_right_idx = _readout_side_indices(brain, id_to_dense, config)

    n_reference_done = 0
    for ref_item in reference_items:
        sentence = reference_sentences[ref_item.reference_id]
        encoded = encode_question_to_input(
            question=sentence.text,
            context="",
            dataset=brain,
            id_to_dense=id_to_dense,
            input_population=config.input.population,
            projection_seed=config.seed.projection,
            balance_seed=config.seed.balance,
            rate_hz_min=config.input.rate_hz_min,
            rate_hz_max=config.input.rate_hz_max,
            n_per_side=config.input.n_per_side,
            model=encoder_model,
        )
        seeds = [
            reference_trial_seed(config.seed.trial_base, ref_item.reference_id, brain, condition, t)
            for t in range(config.trial.n_trials)
        ]
        run_id = ledger.start_reference_run(ref_item.reference_id, brain, condition, cfg_hash, seeds)
        try:
            trials = []
            for t, seed in enumerate(seeds):
                result = net.run_trial((encoded.dense_idx, encoded.rate_hz), duration_ms=config.trial.duration_ms, seed=seed)
                left = int(result.spike_counts[readout_left_idx].sum())
                right = int(result.spike_counts[readout_right_idx].sum())
                trials.append(trial_readout(left, right))
            agg = aggregate_trials(trials, left_is_yes=True, zero_spike_probability=config.readout.zero_spike_probability)
            per_trial_stats = [{"left": t.left, "right": t.right, "bias": t.bias, "zero_spikes": t.zero_spikes} for t in trials]
            ledger.record_reference_prediction(run_id, agg.mean_bias, agg.n_trials, agg.n_zero_spike_trials, per_trial_stats)
            ledger.finish_reference_run(run_id, "done")
            n_reference_done += 1
        except Exception as exc:  # pragma: no cover - defensive, logged and re-raised per item
            logger.exception("reference run failed: %s", ref_item)
            ledger.finish_reference_run(run_id, "failed", error=str(exc))
            raise

    n_done = 0
    if items:
        b0 = handedness.load_b0(ledger, brain, condition, cfg_hash)
        for item in items:
            question = questions[item.question_id]
            text = question.text(item.variant)
            encoded = encode_question_to_input(
                question=text,
                context=question.context,
                dataset=brain,
                id_to_dense=id_to_dense,
                input_population=config.input.population,
                projection_seed=config.seed.projection,
                balance_seed=config.seed.balance,
                rate_hz_min=config.input.rate_hz_min,
                rate_hz_max=config.input.rate_hz_max,
                n_per_side=config.input.n_per_side,
                model=encoder_model,
            )
            seeds = [
                trial_seed(config.seed.trial_base, item.question_id, brain, condition, item.variant, t)
                for t in range(config.trial.n_trials)
            ]
            run_id = ledger.start_run(item.question_id, brain, condition, item.variant, cfg_hash, seeds)
            try:
                trials = []
                for t, seed in enumerate(seeds):
                    result = net.run_trial((encoded.dense_idx, encoded.rate_hz), duration_ms=config.trial.duration_ms, seed=seed)
                    left = int(result.spike_counts[readout_left_idx].sum())
                    right = int(result.spike_counts[readout_right_idx].sum())
                    trials.append(trial_readout(left, right))
                agg = aggregate_trials(trials, left_is_yes=True, zero_spike_probability=config.readout.zero_spike_probability)
                raw_bias = None if agg.all_trials_zero_spikes else agg.mean_bias
                mapping = side_mapping.left_is_yes_for(item.question_id, config.seed.side_base)
                p_yes, corrected = handedness.apply_correction(
                    raw_bias, b0, mapping, config.readout.zero_spike_probability
                )
                per_trial_stats = [{"left": t.left, "right": t.right, "bias": t.bias, "zero_spikes": t.zero_spikes} for t in trials]
                ledger.record_prediction(
                    run_id, p_yes, agg.mean_bias, agg.n_trials, agg.n_zero_spike_trials, per_trial_stats,
                    b0=b0, corrected_bias=corrected, left_is_yes=mapping,
                )
                ledger.finish_run(run_id, "done")
                n_done += 1
            except Exception as exc:  # pragma: no cover - defensive, logged and re-raised per item
                logger.exception("run failed: %s", item)
                ledger.finish_run(run_id, "failed", error=str(exc))
                raise
    return n_done, n_reference_done


def run_no_brain_group(
    brain: str,
    items: list[WorkItem],
    reference_items: list[ReferenceWorkItem],
    questions: dict[str, Question],
    reference_sentences: dict[str, ReferenceSentence],
    config: OracleConfig,
    ledger: Ledger,
    cfg_hash: str,
    encoder_model=None,
) -> tuple[int, int]:
    """No-connectome control for one brain: reference sentences first (so this group's own `b0`
    is defined — README.md "Handedness and side mapping"), then question items, each corrected the
    same way `run_sim_group` corrects its sim-backed items. Returns `(n_question_items_done,
    n_reference_items_done)`."""
    from bioreservoir.oracle.encode import encode_question_to_input
    from bioreservoir.sim import bench

    _n_neurons, _pre_idx, _post_idx, _weight, id_to_dense = bench.graph_to_arrays(brain, min_syn=config.trial.min_syn)

    n_reference_done = 0
    for ref_item in reference_items:
        sentence = reference_sentences[ref_item.reference_id]
        encoded = encode_question_to_input(
            question=sentence.text,
            context="",
            dataset=brain,
            id_to_dense=id_to_dense,
            input_population=config.input.population,
            projection_seed=config.seed.projection,
            balance_seed=config.seed.balance,
            rate_hz_min=config.input.rate_hz_min,
            rate_hz_max=config.input.rate_hz_max,
            n_per_side=config.input.n_per_side,
            model=encoder_model,
        )
        run_id = ledger.start_reference_run(ref_item.reference_id, brain, "no_brain", cfg_hash, trial_seeds=[])
        result = controls.no_brain_baseline(encoded.left_rate_hz, encoded.right_rate_hz)
        raw_bias = result["bias"] if result["bias"] is not None else 0.0
        n_zero = 0 if result["bias"] is not None else 1
        ledger.record_reference_prediction(run_id, raw_bias, n_trials=1, n_zero_spike_trials=n_zero, per_trial_stats=[result])
        ledger.finish_reference_run(run_id, "done")
        n_reference_done += 1

    n_done = 0
    if items:
        b0 = handedness.load_b0(ledger, brain, "no_brain", cfg_hash)
        for item in items:
            question = questions[item.question_id]
            text = question.text(item.variant)
            encoded = encode_question_to_input(
                question=text,
                context=question.context,
                dataset=brain,
                id_to_dense=id_to_dense,
                input_population=config.input.population,
                projection_seed=config.seed.projection,
                balance_seed=config.seed.balance,
                rate_hz_min=config.input.rate_hz_min,
                rate_hz_max=config.input.rate_hz_max,
                n_per_side=config.input.n_per_side,
                model=encoder_model,
            )
            run_id = ledger.start_run(item.question_id, brain, "no_brain", item.variant, cfg_hash, trial_seeds=[])
            result = controls.no_brain_baseline(encoded.left_rate_hz, encoded.right_rate_hz)
            mapping = side_mapping.left_is_yes_for(item.question_id, config.seed.side_base)
            p_yes, corrected = handedness.apply_correction(result["bias"], b0, mapping, zero_spike_probability=0.5)
            ledger.record_prediction(
                run_id, p_yes, result["bias"], n_trials=1, n_zero_spike_trials=(0 if result["bias"] is not None else 1),
                per_trial_stats=[result], b0=b0, corrected_bias=corrected, left_is_yes=mapping,
            )
            ledger.finish_run(run_id, "done")
            n_done += 1
    return n_done, n_reference_done


def run_coin_group(items: list[WorkItem], config: OracleConfig, ledger: Ledger, cfg_hash: str) -> int:
    n_done = 0
    for item in items:
        run_id = ledger.start_run(item.question_id, "n/a", "coin", item.variant, cfg_hash, trial_seeds=[])
        result = controls.coin_control(item.question_id, config.seed.coin_base)
        ledger.record_prediction(run_id, result["p_yes"], mean_bias=None, n_trials=1, n_zero_spike_trials=0, per_trial_stats=[result])
        ledger.finish_run(run_id, "done")
        n_done += 1
    return n_done


def _sim_group_worker(
    brain: str,
    condition: str,
    group_items: list[WorkItem],
    reference_group_items: list[ReferenceWorkItem],
    questions: dict[str, Question],
    reference_sentences: dict[str, ReferenceSentence],
    config: OracleConfig,
    cfg_hash: str,
) -> tuple[int, int]:
    """Top-level, picklable entry point for one `ProcessPoolExecutor` worker: one `(brain,
    condition)` group, its own `Ledger` connection (sqlite3 connections do not survive pickling,
    so each process opens a fresh one against the same on-disk `ledger.sqlite` — SQLite's own
    file locking, not Python, serializes concurrent writers; `Ledger` enables WAL + a busy
    timeout for exactly this)."""
    ledger = Ledger()
    try:
        return run_sim_group(
            brain, condition, group_items, reference_group_items, questions, reference_sentences, config, ledger, cfg_hash
        )
    finally:
        ledger.close()


def run_all(
    question_ids: list[str],
    config: OracleConfig,
    brains: tuple[str, ...] = plan.BRAINS,
    workers: int = 3,
    resume: bool = True,
) -> dict:
    """Build the plan, skip whatever the ledger already has `done` (if `resume`), and execute
    everything else: one `ProcessPoolExecutor` process per `(brain, condition)` sim group (up to
    `workers` at a time — task brief: "--workers N parallel processes (default 3)"), no-brain/coin
    groups directly in this process (cheap, no `LIFNetwork` to build).

    Must run inside the memory/CPU cage (module docstring) — `workers` processes each holding one
    connectome's `LIFNetwork` (~1.6 GB MaleCNS / ~0.8 GB BANC peak RSS, docs/MODEL.md) is exactly
    the scenario the cage budget is sized for (up to 3 processes, see the runbook).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    questions = {q.id: q for q in load_questions() if q.id in question_ids}
    reference_sentences = {r.id: r for r in load_reference(config.handedness.reference_path())}
    ledger = Ledger()
    cfg_hash = compute_config_hash(config)
    for q in questions.values():
        ledger.upsert_question(q.id, q.category, q.resolves_by)
    for r in reference_sentences.values():
        ledger.upsert_reference_sentence(r.id, r.text)

    items = plan.build_plan(list(questions.keys()), brains=brains)
    items = plan.pending_items(items, ledger, cfg_hash) if resume else items
    n_pending = len(items)

    # Reference items are not filtered by `question_ids`/`--question`: every (brain, condition)
    # group needs its own full handedness measurement regardless of which questions this
    # invocation happens to be running (README.md "Handedness and side mapping").
    reference_items = plan.build_reference_plan(list(reference_sentences.keys()), brains=brains)
    reference_items = plan.pending_reference_items(reference_items, ledger, cfg_hash) if resume else reference_items
    n_reference_pending = len(reference_items)

    sim_groups = plan.group_sim_items(items)
    reference_sim_groups = plan.group_reference_sim_items(reference_items)
    no_brain_items = [i for i in items if i.condition == "no_brain"]
    reference_no_brain_items = [i for i in reference_items if i.condition == "no_brain"]
    coin_items = [i for i in items if i.condition == "coin"]
    ledger.close()  # each sim worker opens its own connection; the main process reopens below

    n_done = 0
    n_reference_done = 0
    group_keys = set(sim_groups.keys()) | set(reference_sim_groups.keys())
    if group_keys:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _sim_group_worker,
                    brain,
                    condition,
                    sim_groups.get((brain, condition), []),
                    reference_sim_groups.get((brain, condition), []),
                    questions,
                    reference_sentences,
                    config,
                    cfg_hash,
                ): (brain, condition)
                for brain, condition in group_keys
            }
            for future in as_completed(futures):
                brain, condition = futures[future]
                n, n_ref = future.result()  # re-raises the worker's exception here if it failed
                logger.info(
                    "sim group done: brain=%s condition=%s n_items=%d n_reference_items=%d", brain, condition, n, n_ref
                )
                n_done += n
                n_reference_done += n_ref

    ledger = Ledger()
    for brain in brains:
        brain_items = [i for i in no_brain_items if i.brain == brain]
        brain_ref_items = [i for i in reference_no_brain_items if i.brain == brain]
        if brain_items or brain_ref_items:
            n, n_ref = run_no_brain_group(
                brain, brain_items, brain_ref_items, questions, reference_sentences, config, ledger, cfg_hash
            )
            n_done += n
            n_reference_done += n_ref
    if coin_items:
        n_done += run_coin_group(coin_items, config, ledger, cfg_hash)
    ledger.close()

    return {
        "n_work_items_pending": n_pending,
        "n_work_items_run": n_done,
        "n_reference_items_pending": n_reference_pending,
        "n_reference_items_run": n_reference_done,
    }
