"""Work-item enumeration, ledger-aware filtering (resumability) and the `--dry-run` cost estimate.

One full batch = every question x every (brain, condition, text variant) combination, plus the two
cheap controls (no-brain, coin) that do not need a connectome at all (task brief item 6). The
sim-backed combinations (`condition in {real, rewired, er}`) are the only ones that cost real CPU
time; `dry_run_report` reuses `bioreservoir.sim.calibrate.throughput` (not reimplemented) for that
estimate, fed with the measured s-wall/s-simulated figures already committed in docs/MODEL.md's CPU
benchmark table.
"""

from __future__ import annotations

from dataclasses import dataclass

from bioreservoir.oracle.config import GRAPH_VARIANTS, TEXT_VARIANTS, OracleConfig
from bioreservoir.oracle.ledger import Ledger

BRAINS = ("malecns", "banc")

# docs/MODEL.md "CPU benchmark" table: s wall / s simulated, cage (MemoryMax=6G, CPUQuota=400%,
# single process), min_syn=5, dt=0.1 ms (Brian2 default), cython codegen. Used only as the
# `--dry-run` estimate's assumption, restated explicitly in its own output (never silently
# implied) since it is a measured constant from a *different* run, not something this module
# reproduces.
MEASURED_SEC_WALL_PER_SEC_SIMULATED = {"malecns": 27.48, "banc": 18.98}

SIM_CONDITIONS = GRAPH_VARIANTS  # ("real", "rewired", "er") — the three that need a LIFNetwork
CONTROL_CONDITIONS = ("no_brain", "coin")

# Handedness reference sentences (oracle.reference, oracle.handedness) run through every condition
# a question's raw bias needs a b0 for: the three sim conditions plus the no-brain baseline. Not
# "coin" — a coin flip has no lateral-bias concept to correct at all.
REFERENCE_CONDITIONS = (*SIM_CONDITIONS, "no_brain")


@dataclass(frozen=True)
class WorkItem:
    """One (question, brain, condition, variant) combination the runner must eventually cover.

    `brain` is `"n/a"` for `condition == "coin"` (brain-independent, README.md Pipeline step 4d);
    `variant` is always `"original"` for `condition == "coin"` too — a physical coin flip does not
    depend on how the question is worded, so one flip is reused for every text variant rather than
    stored three times (see `oracle.plan.build_plan`'s docstring for the reasoning).
    """

    question_id: str
    brain: str
    condition: str
    variant: str


def build_plan(
    question_ids: list[str],
    brains: tuple[str, ...] = BRAINS,
    sim_conditions: tuple[str, ...] = SIM_CONDITIONS,
    text_variants: tuple[str, ...] = TEXT_VARIANTS,
) -> list[WorkItem]:
    items: list[WorkItem] = []
    for qid in question_ids:
        for brain in brains:
            for condition in sim_conditions:
                for variant in text_variants:
                    items.append(WorkItem(qid, brain, condition, variant))
            for variant in text_variants:
                items.append(WorkItem(qid, brain, "no_brain", variant))
        items.append(WorkItem(qid, "n/a", "coin", "original"))
    return items


def pending_items(items: list[WorkItem], ledger: Ledger, config_hash: str) -> list[WorkItem]:
    """`items` minus whatever `ledger` already has `done` for this exact `config_hash` — a config
    change (any seed, trial count, readout, ...) reruns everything under the new hash rather than
    silently mixing results from two different configurations (oracle.config.config_hash)."""
    return [
        item
        for item in items
        if not ledger.is_done(item.question_id, item.brain, item.condition, item.variant, config_hash)
    ]


def group_sim_items(items: list[WorkItem]) -> dict[tuple[str, str], list[WorkItem]]:
    """Sim-backed items (`condition in SIM_CONDITIONS`) grouped by `(brain, condition)` — one
    `LIFNetwork` build serves everything in one group (task brief: "One network build per (brain,
    condition) per worker, reused via store/restore.")."""
    groups: dict[tuple[str, str], list[WorkItem]] = {}
    for item in items:
        if item.condition not in SIM_CONDITIONS:
            continue
        groups.setdefault((item.brain, item.condition), []).append(item)
    return groups


@dataclass(frozen=True)
class ReferenceWorkItem:
    """One `(reference_id, brain, condition)` combination the runner must cover before any
    question in that `(brain, condition)` group can be corrected for handedness (`oracle.
    handedness`). No `variant` — reference sentences have no negation/paraphrase — and no `coin`
    condition (`REFERENCE_CONDITIONS`)."""

    reference_id: str
    brain: str
    condition: str


def build_reference_plan(
    reference_ids: list[str],
    brains: tuple[str, ...] = BRAINS,
    conditions: tuple[str, ...] = REFERENCE_CONDITIONS,
) -> list[ReferenceWorkItem]:
    return [
        ReferenceWorkItem(rid, brain, condition)
        for rid in reference_ids
        for brain in brains
        for condition in conditions
    ]


def pending_reference_items(
    items: list[ReferenceWorkItem], ledger: Ledger, config_hash: str
) -> list[ReferenceWorkItem]:
    """Same resume rule as `pending_items`, against `Ledger.is_reference_done`."""
    return [
        item
        for item in items
        if not ledger.is_reference_done(item.reference_id, item.brain, item.condition, config_hash)
    ]


def group_reference_sim_items(
    items: list[ReferenceWorkItem],
) -> dict[tuple[str, str], list[ReferenceWorkItem]]:
    """Sim-backed reference items (`condition in SIM_CONDITIONS`) grouped by `(brain, condition)` —
    processed in the same worker, against the same `LIFNetwork` build, as that group's question
    items (`oracle.runner.run_sim_group`); the `no_brain` reference items need no connectome and
    are handled directly, like their question counterparts."""
    groups: dict[tuple[str, str], list[ReferenceWorkItem]] = {}
    for item in items:
        if item.condition not in SIM_CONDITIONS:
            continue
        groups.setdefault((item.brain, item.condition), []).append(item)
    return groups


def dry_run_report(
    question_ids: list[str],
    config: OracleConfig,
    brains: tuple[str, ...] = BRAINS,
    n_parallel: int | None = None,
    sec_wall_per_sec_simulated: dict[str, float] = MEASURED_SEC_WALL_PER_SEC_SIMULATED,
    reference_ids: list[str] | None = None,
) -> dict:
    """The work plan plus an estimated CPU-hours figure for the sim-backed combinations, via
    `bioreservoir.sim.calibrate.throughput` (reused, see module docstring) — for both the question
    batch and the handedness reference batch (`oracle.reference`, default: every sentence in
    `config.handedness.reference_file`), reported separately and combined, since the reference
    batch must run at least once per `(brain, condition)` before any question in that group has a
    b0 to be corrected against (`oracle.handedness`)."""
    from bioreservoir.sim.calibrate import throughput

    if reference_ids is None:
        from bioreservoir.oracle.reference import load_reference

        reference_ids = [r.id for r in load_reference(config.handedness.reference_path())]

    items = build_plan(question_ids, brains=brains)
    sim_groups = group_sim_items(items)
    reference_items = build_reference_plan(reference_ids, brains=brains)
    reference_sim_groups = group_reference_sim_items(reference_items)
    n_parallel = n_parallel or config.workers.default

    per_trial_wall_s = {
        brain: sec_wall_per_sec_simulated[brain] * (config.trial.duration_ms / 1000.0) for brain in brains
    }
    cost = throughput(
        per_trial_wall_s=per_trial_wall_s,
        n_brains=len(brains),
        n_graph_variants=len(SIM_CONDITIONS),
        n_text_variants=len(TEXT_VARIANTS),
        n_trials_per_condition=config.trial.n_trials,
        n_questions=len(question_ids),
        n_parallel=n_parallel,
    )
    # Reference sentences have no text variant (n_text_variants=1) — one run per sentence per
    # (brain, sim condition), not three (`REFERENCE_CONDITIONS`'s `no_brain` entry is free, same
    # as the question plan's own no-brain items, and is not part of this sim-only cost estimate).
    reference_cost = throughput(
        per_trial_wall_s=per_trial_wall_s,
        n_brains=len(brains),
        n_graph_variants=len(SIM_CONDITIONS),
        n_text_variants=1,
        n_trials_per_condition=config.trial.n_trials,
        n_questions=len(reference_ids),
        n_parallel=n_parallel,
    )
    n_no_brain = sum(1 for i in items if i.condition == "no_brain")
    n_coin = sum(1 for i in items if i.condition == "coin")
    n_reference_no_brain = sum(1 for i in reference_items if i.condition == "no_brain")
    return {
        "n_questions": len(question_ids),
        "n_work_items": len(items),
        "n_sim_groups": len(sim_groups),
        "sim_groups": sorted(sim_groups.keys()),
        "n_no_brain_items": n_no_brain,
        "n_coin_items": n_coin,
        "assumed_sec_wall_per_sec_simulated": sec_wall_per_sec_simulated,
        "trial_duration_ms": config.trial.duration_ms,
        "n_trials_per_condition": config.trial.n_trials,
        "cost_estimate": cost,
        "n_reference_sentences": len(reference_ids),
        "n_reference_work_items": len(reference_items),
        "n_reference_sim_groups": len(reference_sim_groups),
        "n_reference_no_brain_items": n_reference_no_brain,
        "reference_cost_estimate": reference_cost,
        "combined_cost_estimate": {
            "total_trials": cost["total_trials"] + reference_cost["total_trials"],
            "sequential_hours": cost["sequential_hours"] + reference_cost["sequential_hours"],
            "parallel_hours": cost["parallel_hours"] + reference_cost["parallel_hours"],
        },
    }
