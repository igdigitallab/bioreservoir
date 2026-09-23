"""`python -m bioreservoir.live.validate_states` — proposes `validated: true/false` +
`normalization` ranges for `experiments/001-fly-oracle/live-states.yaml` (see that file's
`validation_procedure` for the exact method this module implements).

Reuses `worker.build_resources()` (same network build, encoder, population indices as the live
worker itself) rather than duplicating any of it.

Must run inside the memory/CPU cage described in experiments/001-fly-oracle/RUNNING.md (same convention as
`oracle.runner`/`sim.bench`/`live.worker`) — this module does not build the cage itself, it
assumes it is already inside one.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from bioreservoir.live import config, states
from bioreservoir.oracle import reference
from bioreservoir.oracle.seeding import stable_seed

logger = logging.getLogger(__name__)

VARIED_YAML = config.EXPERIMENT_DIR / "live-varied.yaml"

# Distinct seed namespace from every other trial-seed function in this codebase (`oracle.seeding`,
# `live.worker`'s own `LIVE_TRIAL_SEED_BASE`) — state-validation trials are their own thing.
VALIDATE_SEED_BASE = 20260918779

# Across-text std >= DEFAULT_THRESHOLD x mean within-text std -> propose `validated: true` (see
# live-states.yaml's `validation_procedure` for the reasoning).
DEFAULT_THRESHOLD = 2.0


def load_varied(path: Path = VARIED_YAML) -> list[reference.ReferenceSentence]:
    """The 24 varied yes/no questions (task brief). Committed alongside `reference.yaml`, not
    generated at runtime — this script reads, it does not invent evaluation content on its own."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — expected a committed 24-question set (see live-states.yaml's "
            "validation_procedure and this repo's experiments/001-fly-oracle/live-varied.yaml)."
        )
    raw = yaml.safe_load(path.read_text())
    return [reference.ReferenceSentence(id=row["id"], text=row["text"]) for row in raw]


@dataclass(frozen=True)
class TextStateReading:
    """One text's `cfg.trial.n_trials` raw readings for one state (spike count, or the
    active-neuron fraction for `arousal`)."""

    text_id: str
    trial_values: list[float]

    @property
    def mean(self) -> float:
        return float(np.mean(self.trial_values))

    @property
    def std(self) -> float:
        return float(np.std(self.trial_values))


def run_state_readings(
    net,
    id_to_dense: dict[int, int],
    cfg,
    texts: list,
    state_specs: dict[str, states.StateSpec],
    state_population_idx: dict[str, np.ndarray | None],
    encoder_model=None,
) -> dict[str, list[TextStateReading]]:
    """`{state_name: [TextStateReading, ...]}`, one entry per text in `texts`. `net` is
    duck-typed (same convention as `worker.answer_question`: only needs `.run_trial(...)` ->
    something with `.spike_counts`), so this is unit-testable with a fake network."""
    from bioreservoir.oracle.encode import encode_question_to_input

    n_total = len(id_to_dense)
    readings: dict[str, list[TextStateReading]] = {name: [] for name in states.STATE_NAMES}
    for text in texts:
        encoded = encode_question_to_input(
            question=text.text,
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
        per_state_trials: dict[str, list[float]] = {name: [] for name in states.STATE_NAMES}
        for t in range(cfg.trial.n_trials):
            seed = stable_seed(text.id, str(t), base=VALIDATE_SEED_BASE) % 2**32
            result = net.run_trial(
                (encoded.dense_idx, encoded.rate_hz), duration_ms=cfg.trial.duration_ms, seed=seed
            )
            spike_counts = np.asarray(result.spike_counts)
            for name in states.STATE_NAMES:
                spec = state_specs[name]
                if name == "arousal":
                    value = states.raw_arousal(int((spike_counts > 0).sum()), n_total)
                else:
                    counts = {
                        pop: (
                            int(spike_counts[state_population_idx[pop]].sum())
                            if state_population_idx.get(pop) is not None and state_population_idx[pop].size
                            else 0
                        )
                        for pop in spec.populations
                    }
                    value = float(states.raw_appetite_fear_backoff_courtship(counts, spec))
                per_state_trials[name].append(value)
        for name in states.STATE_NAMES:
            readings[name].append(TextStateReading(text_id=text.id, trial_values=per_state_trials[name]))
    return readings


def propose_validation(
    readings: list[TextStateReading], threshold: float = DEFAULT_THRESHOLD
) -> tuple[bool, dict, dict]:
    """`(validated, normalization, stats)` for one state from its `readings` across every text —
    live-states.yaml's `validation_procedure`: across-text std of the per-text means vs. the mean
    within-text (trial-to-trial) std."""
    means = np.array([r.mean for r in readings])
    within_stds = np.array([r.std for r in readings])
    across_std = float(np.std(means))
    mean_within_std = float(np.mean(within_stds))
    validated = across_std >= threshold * mean_within_std if mean_within_std > 0 else across_std > 0
    lo, hi = float(np.percentile(means, 5)), float(np.percentile(means, 95))
    stats = {"across_text_std": across_std, "mean_within_text_std": mean_within_std, "n_texts": len(readings)}
    return validated, {"min_spikes": lo, "max_spikes": hi}, stats


def write_proposals(path: Path, proposals: dict[str, tuple[bool, dict, dict]]) -> None:
    """Updates `validated`/`normalization` in place for every state in `proposals` — never
    touches `arousal` (not in `proposals`, `main()` below never computes one for it, see
    live-states.yaml's own note for why it needs no calibration)."""
    raw = yaml.safe_load(path.read_text())
    for name, (validated, normalization, _stats) in proposals.items():
        raw["states"][name]["validated"] = validated
        raw["states"][name]["normalization"] = normalization
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the report only, do not write live-states.yaml"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from bioreservoir.live.worker import build_resources

    resources = build_resources()
    texts = list(reference.load_reference(resources["cfg"].handedness.reference_path())) + list(load_varied())
    logger.info(
        "running %d texts (24 reference + 24 varied) x %d trials for state validation",
        len(texts), resources["cfg"].trial.n_trials,
    )

    readings = run_state_readings(
        resources["net"],
        resources["id_to_dense"],
        resources["cfg"],
        texts,
        resources["state_specs"],
        resources["state_population_idx"],
        resources["encoder_model"],
    )

    proposals: dict[str, tuple[bool, dict, dict]] = {}
    for name in states.STATE_NAMES:
        if name == "arousal":
            continue  # never re-validated, see live-states.yaml's own note
        validated, normalization, stats = propose_validation(readings[name], threshold=args.threshold)
        proposals[name] = (validated, normalization, stats)
        print(f"{name}: validated={validated} normalization={normalization} stats={stats}")

    if not args.dry_run:
        write_proposals(config.LIVE_STATES_YAML, proposals)
        print(f"\nwrote proposals into {config.LIVE_STATES_YAML} — review before committing.")
    else:
        print("\n--dry-run: live-states.yaml not written.")


if __name__ == "__main__":
    main()
