"""`python -m bioreservoir.live.reproduce --id <id> --question "<text>"` — recomputes one live
answer from scratch, independent of the SQLite store (task brief: "a CLI that recomputes one
answer from its seed/config"; `Answer.lab.provenance.reproduce` is exactly this command).

Determinism is structural, not incidental: every seed in the live pipeline
(`worker.live_trial_seed`, `oracle.side_mapping.left_is_yes_for`) is derived only from the normalized `--question` text
(`pipeline.question_key`) and `config.LIVE_TRIAL_SEED_BASE`/`config.LIVE_SIDE_BASE`, and the encoder/network are built the same
way `worker.build_resources` builds them from the same committed `config.yaml` — so the SAME
`--id`/`--question` pair always reproduces the SAME `Answer`, byte for byte, as long as the
underlying data/config/code have not changed (which is exactly what `provenance.config_hash`/
`code_sha` let a reader verify).

Must run inside the memory/CPU cage described in experiments/001-fly-oracle/RUNNING.md (`worker.build_resources` builds a
real `LIFNetwork`) — this module does not build the cage itself, it assumes it is already inside
one, same rule as `worker.py`/`validate_states.py`.
"""

from __future__ import annotations

import argparse
import json
import logging

from bioreservoir.live import worker

logger = logging.getLogger(__name__)


def reproduce_answer(id_: int, question: str, resources: dict) -> dict:
    """Recompute one Answer from `resources` (the exact shape `worker.build_resources()`
    returns) — pure glue over `worker.answer_question`, kept as its own function so `main()`'s
    CLI and this module's own determinism test share one code path rather than duplicating it.

    Task brief: "update reproduce.py so the reproduce command recomputes all three contenders" —
    `resources.get(...)` (not `resources[...]`) so a caller/test whose `resources` dict predates
    the game feature (no `er_pool`/`random_graph_b0`/`no_brain_b0` keys) still reproduces the
    real brain's own answer exactly as before, with `Answer.game` simply absent (same convention
    as `worker.answer_question`'s own `er_pool=None` default).
    """
    row = {"id": id_, "question": question}
    return worker.answer_question(
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, required=True, help="the live queue row id to reproduce")
    parser.add_argument("--question", type=str, required=True, help="the exact question text that was asked")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger.info("reproducing answer id=%d question=%r", args.id, args.question)

    resources = worker.build_resources()
    try:
        answer = reproduce_answer(args.id, args.question, resources)
        print(json.dumps(answer, indent=2, sort_keys=True))
    finally:
        # None if build_resources() disabled the game at startup (missing/broken ER cache or pool).
        if resources["er_pool"] is not None:
            resources["er_pool"].shutdown()


if __name__ == "__main__":
    main()
